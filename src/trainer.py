import os
import os.path as osp
import json
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from shutil import copyfile
import numpy as np

from .dataset import TIGREDataset
from .network import sdf_freq_mlp, att_freq_mlp, sdf_hash_mlp, att_hash_mlp
from .render import render_image, surface_boundary_function, volume_render_intensity
from .utils import get_psnr, get_mse, get_psnr_3d, get_ssim_3d, cast_to_image


class Trainer:
    def __init__(self, cfg, device="cuda"):
        """Initialize NeAS trainer.
        
        Args:
            cfg: Configuration dictionary
            device: Device to run training on
        """
        # Args
        self.global_step = 0
        self.conf = cfg
        self.epochs = cfg["train"]["epoch"]
        self.i_eval = cfg["log"]["i_eval"]
        self.i_save = cfg["log"]["i_save"]
        self.n_rays = cfg["train"]["n_rays"]
        self.n_samples = cfg["train"]["n_samples"]
        self.val_n_samples = cfg["train"]["val_n_samples"]
        self.val_chunk_size = cfg["train"]["val_chunk_size"]
        self.lambda_reg = cfg["train"]["lambda_reg"]
        self.device = device
        
        # Frequency regularization (coarse-to-fine)
        self.use_freq_reg = cfg["train"].get("use_freq_reg", False)
        self.tau_start = cfg["train"].get("tau_start", 2.0)
        self.warmup_iters = cfg["train"].get("warmup_iters", 500)
        
        # Pose refinement
        self.use_pose_refinement = cfg["train"].get("use_pose_refinement", False)
        self.pose_warmup_iters = cfg["train"].get("pose_warmup_iters", 500)
  
        # Log directory
        self.expdir = osp.join(cfg["exp"]["expdir"], cfg["exp"]["expname"])
        self.ckptdir = osp.join(self.expdir, "ckpt.tar")
        self.ckptdir_backup = osp.join(self.expdir, "ckpt_backup.tar")
        self.evaldir = osp.join(self.expdir, "eval")
        os.makedirs(self.expdir, exist_ok=True)
        os.makedirs(self.evaldir, exist_ok=True)

        # Dataset
        train_dset = TIGREDataset(cfg["exp"]["datadir"], cfg["train"]["n_rays"], "train", device)
        self.eval_dset = TIGREDataset(cfg["exp"]["datadir"], cfg["train"]["n_rays"], "val", device) if self.i_eval > 0 else None
        self.train_dloader = torch.utils.data.DataLoader(train_dset, batch_size=cfg["train"]["n_batch"], shuffle=True)
        
        self.geo = train_dset.geo
        self.n_train_images = train_dset.n_samples
    
        # Network
        feature_dim = cfg["network"].get("feature_dim", 8)
        multires = cfg["network"].get("multires", 6)
        encoding_type = cfg["network"].get("encoding_type", "freq")  # 'freq' or 'hash'
        
        alpha = cfg["network"].get("alpha", 3.4)
        beta = cfg["network"].get("beta", 0.1)
        
        if encoding_type == "freq":
            self.sdf_model = sdf_freq_mlp(input_dim=3, output_dim=1, feature_dim=feature_dim, multires=multires).to(device)
            self.att_model = att_freq_mlp(input_dim=feature_dim, output_dim=1, multires=multires, alpha=alpha, beta=beta).to(device)
        elif encoding_type == "hash":
            num_levels = cfg["network"].get("num_levels", 14)
            level_dim = cfg["network"].get("level_dim", 2)
            base_resolution = cfg["network"].get("base_resolution", 16)
            log2_hashmap_size = cfg["network"].get("log2_hashmap_size", 19)
            
            self.sdf_model = sdf_hash_mlp(input_dim=3, output_dim=1, feature_dim=feature_dim,
                                         num_levels=num_levels, level_dim=level_dim,
                                         base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size).to(device)
            self.att_model = att_hash_mlp(input_dim=feature_dim, output_dim=1,
                                         num_levels=num_levels, level_dim=level_dim,
                                         base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size,
                                         alpha=alpha, beta=beta).to(device)
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
        
        self.encoding_type = encoding_type
        
        # Boundary sharpness parameter (learnable)
        s_param = cfg["network"].get("s_param", 20.0)
        self.s = torch.nn.Parameter(torch.tensor(s_param, device=device))
        
        if self.use_pose_refinement:
            self.pose_trans_delta = torch.nn.Parameter(torch.zeros(self.n_train_images, 3, device=device))
            self.principal_point_delta = torch.nn.Parameter(torch.zeros(self.n_train_images, 2, device=device))
        else:
            self.pose_trans_delta = None
            self.principal_point_delta = None
        
        # Optimizer
        grad_vars = list(self.sdf_model.parameters()) + list(self.att_model.parameters()) + [self.s]
        
        if self.use_pose_refinement:
            grad_vars += [self.pose_trans_delta, self.principal_point_delta]
        
        self.optimizer = torch.optim.Adam(params=grad_vars, lr=cfg["train"]["lrate"])
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self.optimizer, 
            step_size=cfg["train"]["lrate_step"], 
            gamma=cfg["train"]["lrate_gamma"]
        )

        # Load checkpoints
        self.epoch_start = 0
        if cfg["train"].get("resume", False) and osp.exists(self.ckptdir):
            print(f"Load checkpoints from {self.ckptdir}.")
            ckpt = torch.load(self.ckptdir)
            self.epoch_start = ckpt["epoch"] + 1
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.global_step = self.epoch_start * len(self.train_dloader)
            self.sdf_model.load_state_dict(ckpt["sdf_model_state_dict"])
            self.att_model.load_state_dict(ckpt["att_model_state_dict"])
            self.s.data = ckpt["s"]
            if self.use_pose_refinement and "pose_trans_delta" in ckpt:
                self.pose_trans_delta.data = ckpt["pose_trans_delta"]
                self.principal_point_delta.data = ckpt["principal_point_delta"]

        # Summary writer
        self.writer = SummaryWriter(self.expdir)
        self.writer.add_text("parameters", self.args2string(cfg), global_step=0)
        
        # Loss history
        self.loss_history = {'total': [], 'int': [], 'reg': []}

    def args2string(self, hp):
        """Transfer args to string."""
        json_hp = json.dumps(hp, indent=2)
        return "".join("\t" + line for line in json_hp.splitlines(True))

    def compute_loss(self, batch_data, global_step):
        """Compute loss for a batch.
        
        Args:
            batch_data: Dictionary with 'rays' and 'projs'
            global_step: Current global iteration
            
        Returns:
            Dictionary with losses
        """
        rays = batch_data['rays'].to(self.device)
        projs = batch_data['projs'].to(self.device)
        
        ray_origins = rays[..., :3]
        ray_directions = rays[..., 3:6] 
        near = rays[..., 6:7]
        far = rays[..., 7:8]
        
        batch_size, n_rays, _ = ray_origins.shape
        
        # Coarse-to-fine frequency regularization (only for frequency encoding)
        tau = None
        if self.use_freq_reg and self.encoding_type == 'freq':
            if global_step < self.warmup_iters:
                tau = None
            else:
                iters_after_warmup = global_step - self.warmup_iters
                total_iters = self.epochs * len(self.train_dloader)
                half_iters = total_iters // 2
                effective_half_iters = half_iters - self.warmup_iters
                
                max_freq_L = self.sdf_model.encoder.N_freqs
                
                if iters_after_warmup < effective_half_iters:
                    progress = iters_after_warmup / effective_half_iters
                    tau = self.tau_start + progress * (max_freq_L - self.tau_start)
                else:
                    tau = float(max_freq_L)
        
        # Ray point sampling
        t_vals = torch.linspace(0., 1., steps=self.n_samples, device=self.device)
        sample_depths = near * (1. - t_vals) + far * t_vals
        sampled_points = ray_origins.unsqueeze(2) + ray_directions.unsqueeze(2) * sample_depths.unsqueeze(-1)
        sampled_points_flat = sampled_points.reshape(-1, 3)
        
        # Forward pass
        sdf_distances, feature_vector = self.sdf_model(sampled_points_flat, tau=tau)
        boundary_values = surface_boundary_function(sdf_distances, self.s)
        attenuation_values = self.att_model(feature_vector)
        
        # Attenuation coefficient
        att_coeff = attenuation_values.squeeze(-1) * boundary_values
        att_coeff = att_coeff.reshape(batch_size, n_rays, self.n_samples)
        
        # Volume rendering
        dists = (far - near) / self.n_samples
        pred_intensity = volume_render_intensity(att_coeff, dists)
        gt_intensity = torch.exp(-projs)
        
        # Intensity loss
        L_int = torch.nn.functional.mse_loss(pred_intensity, gt_intensity)
        
        # Eikonal regularization
        pts_eikonal = sampled_points_flat.clone().detach().requires_grad_(True)
        sdf_eikonal, _ = self.sdf_model(pts_eikonal, tau=tau)
        sdf_sum = sdf_eikonal.sum()
        
        n = torch.autograd.grad(
            outputs=sdf_sum,
            inputs=pts_eikonal,
            create_graph=True,
        )[0]
        
        n_norm = torch.linalg.norm(n, dim=-1)
        L_reg = torch.mean((n_norm - 1.0)**2)
        
        # Total loss
        L_total = L_int + self.lambda_reg * L_reg
        
        loss = {
            "loss": L_total,
            "loss_int": L_int,
            "loss_reg": L_reg
        }
        
        # Log
        self.writer.add_scalar("train/loss", L_total.item(), global_step)
        self.writer.add_scalar("train/loss_int", L_int.item(), global_step)
        self.writer.add_scalar("train/loss_reg", L_reg.item(), global_step)
        
        return loss

    def eval_step(self, global_step, idx_epoch):
        """Evaluation step."""
        print("Running validation...")
        self.sdf_model.eval()
        self.att_model.eval()
        
        val_save_dir = osp.join(self.expdir, f'val_epoch_{idx_epoch}')
        os.makedirs(val_save_dir, exist_ok=True)

        with torch.no_grad():
            # Random projection
            select_ind = np.random.choice(len(self.eval_dset))
            rays = self.eval_dset.rays[select_ind].to(self.device)
            projs = self.eval_dset.projs[select_ind].to(self.device)
            
            img = render_image(rays, self.sdf_model, self.att_model, self.s, 
                             self.val_n_samples, chunk_size=self.val_chunk_size, tau=None)

            # Visualization
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(img.cpu().numpy().T, cmap='gray')
            plt.title('Predicted')
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(torch.exp(-projs).cpu().numpy().T, cmap='gray')
            plt.title('Ground Truth')
            plt.axis('off')
            plt.savefig(os.path.join(val_save_dir, f'val_{select_ind}.png'))
            plt.close()
            
            # Compute metrics
            proj_mse = get_mse(img, torch.exp(-projs))
            proj_psnr = get_psnr(img, torch.exp(-projs))
            
            self.writer.add_scalar("eval/proj_mse", proj_mse, global_step)
            self.writer.add_scalar("eval/proj_psnr", proj_psnr, global_step)
            self.writer.add_image("eval/proj_pred", cast_to_image(img.cpu().numpy().T), global_step, dataformats="HWC")
            
        self.sdf_model.train()
        self.att_model.train()

    def start(self):
        """Main training loop."""
        iter_per_epoch = len(self.train_dloader)
        pbar = tqdm(total=iter_per_epoch * self.epochs, leave=True)
        if self.epoch_start > 0:
            pbar.update(self.epoch_start * iter_per_epoch)

        for idx_epoch in range(self.epoch_start, self.epochs + 1):
            
            # Evaluate
            if (idx_epoch % self.i_eval == 0 or idx_epoch == self.epochs) and self.i_eval > 0 and idx_epoch > 0:
                self.eval_step(global_step=self.global_step, idx_epoch=idx_epoch)
            
            # Save checkpoint
            if (idx_epoch % self.i_save == 0 or idx_epoch == self.epochs) and idx_epoch > 0:
                save_path = osp.join(self.expdir, f'checkpoint_epoch_{idx_epoch}.pth')
                checkpoint_dict = {
                    'epoch': idx_epoch,
                    'args': self.conf,
                    'sdf_model_state_dict': self.sdf_model.state_dict(),
                    'att_model_state_dict': self.att_model.state_dict(),
                    's': self.s.data,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.lr_scheduler.state_dict(),
                    'loss_history': self.loss_history,
                    'feature_dim': self.conf["network"].get("feature_dim", 8),
                    'encoding_type': self.encoding_type,
                    'alpha': self.conf["network"].get("alpha", 3.4),
                    'beta': self.conf["network"].get("beta", 0.1),
                }
                
                # Save hash encoding parameters if using hash encoding
                if self.encoding_type == 'hash':
                    checkpoint_dict['num_levels'] = self.conf["network"].get("num_levels", 14)
                    checkpoint_dict['level_dim'] = self.conf["network"].get("level_dim", 2)
                    checkpoint_dict['base_resolution'] = self.conf["network"].get("base_resolution", 16)
                    checkpoint_dict['log2_hashmap_size'] = self.conf["network"].get("log2_hashmap_size", 19)
                
                # Save pose refinement parameters if enabled
                if self.use_pose_refinement:
                    checkpoint_dict['pose_trans_delta'] = self.pose_trans_delta.data
                    checkpoint_dict['principal_point_delta'] = self.principal_point_delta.data
                
                torch.save(checkpoint_dict, save_path)
                print(f"Checkpoint saved at epoch {idx_epoch}")
            
            if idx_epoch == self.epochs:
                break
                
            # Training
            epoch_loss = 0.0
            epoch_int_loss = 0.0
            epoch_reg_loss = 0.0
            
            for batch_idx, batch_data in enumerate(self.train_dloader):
                self.optimizer.zero_grad()
                
                loss = self.compute_loss(batch_data, self.global_step)
                loss["loss"].backward()
                self.optimizer.step()
                
                epoch_int_loss += loss["loss_int"].item()
                epoch_reg_loss += loss["loss_reg"].item()
                epoch_loss += loss["loss"].item()
                
                self.global_step += 1
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f"{loss['loss'].item():.4f}",
                    'int': f"{loss['loss_int'].item():.4f}",
                    'reg': f"{loss['loss_reg'].item():.4f}"
                })
            
            # Average losses
            avg_loss = epoch_loss / len(self.train_dloader)
            avg_int_loss = epoch_int_loss / len(self.train_dloader)
            avg_reg_loss = epoch_reg_loss / len(self.train_dloader)
            
            self.loss_history['total'].append(avg_loss)
            self.loss_history['int'].append(avg_int_loss)
            self.loss_history['reg'].append(self.lambda_reg * avg_reg_loss)
            
            # Plot loss curve
            plt.figure(figsize=(10, 5))
            plt.plot(self.loss_history['total'], label='Total Loss')
            plt.plot(self.loss_history['int'], label='Intensity Loss')
            plt.plot(self.loss_history['reg'], label='Regularization Loss (Lambda scaled)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.expdir, 'loss_curve.png'))
            plt.close()
            
            self.lr_scheduler.step()

        pbar.close()
        print("Training completed!")
