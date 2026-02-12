import os
import os.path as osp
import json
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from shutil import copyfile
import numpy as np
import wandb

from .dataset import TIGREDataset
from .network import sdf_freq_mlp, att_freq_mlp, sdf_hash_mlp, att_hash_mlp, sdf_freq_mlp_2m, sdf_hash_mlp_2m, selector_function
from .render import render_image, surface_boundary_function, volume_render_intensity
from .utils import get_psnr, get_mse, get_psnr_3d, get_ssim, get_ssim_3d, cast_to_image


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
        self.perturb = cfg["train"].get("perturb", True)
        
        self.use_wandb = cfg["log"].get("use_wandb", False)
        self.wandb_project = cfg["log"].get("wandb_project", "neas")
        self.wandb_entity = cfg["log"].get("wandb_entity", None)
  
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
        num_materials = cfg["network"].get("num_materials", 1)  # 1 for 1M-NeAS, 2 for 2M-NeAS
        
        # Activation function parameters for material 1
        alpha1 = cfg["network"].get("alpha1", cfg["network"].get("alpha", 3.4))
        beta1 = cfg["network"].get("beta1", cfg["network"].get("beta", 0.1))
        
        # Activation function parameters for material 2 (only for 2M-NeAS)
        alpha2 = cfg["network"].get("alpha2", 5.5)
        beta2 = cfg["network"].get("beta2", 3.5)
        
        self.num_materials = num_materials
        
        # Create SDF network (single or dual output)
        if encoding_type == "freq":
            if num_materials == 1:
                self.sdf_model = sdf_freq_mlp(input_dim=3, output_dim=1, feature_dim=feature_dim, multires=multires).to(device)
            else:
                self.sdf_model = sdf_freq_mlp_2m(input_dim=3, output_dim=2, feature_dim=feature_dim, multires=multires).to(device)
            
            self.att_model1 = att_freq_mlp(input_dim=feature_dim, output_dim=1, alpha=alpha1, beta=beta1).to(device)
            if num_materials == 2:
                self.att_model2 = att_freq_mlp(input_dim=feature_dim, output_dim=1, alpha=alpha2, beta=beta2).to(device)
            else:
                self.att_model2 = None
                
        elif encoding_type == "hash":
            num_levels = cfg["network"].get("num_levels", 14)
            level_dim = cfg["network"].get("level_dim", 2)
            base_resolution = cfg["network"].get("base_resolution", 16)
            log2_hashmap_size = cfg["network"].get("log2_hashmap_size", 19)
            
            if num_materials == 1:
                self.sdf_model = sdf_hash_mlp(input_dim=3, output_dim=1, feature_dim=feature_dim,
                                             num_levels=num_levels, level_dim=level_dim,
                                             base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size).to(device)
            else:  # num_materials == 2
                self.sdf_model = sdf_hash_mlp_2m(input_dim=3, output_dim=2, feature_dim=feature_dim,
                                                num_levels=num_levels, level_dim=level_dim,
                                                base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size).to(device)
            
            # Create attenuation network(s)
            self.att_model1 = att_hash_mlp(input_dim=feature_dim, output_dim=1,
                                          num_levels=num_levels, level_dim=level_dim,
                                          base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size,
                                          alpha=alpha1, beta=beta1).to(device)
            if num_materials == 2:
                self.att_model2 = att_hash_mlp(input_dim=feature_dim, output_dim=1,
                                              num_levels=num_levels, level_dim=level_dim,
                                              base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size,
                                              alpha=alpha2, beta=beta2).to(device)
            else:
                self.att_model2 = None
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
        
        self.encoding_type = encoding_type
        
        # Boundary sharpness parameter (learnable)
        s_param = cfg["network"].get("s_param", 20.0)
        self.s = torch.nn.Parameter(torch.tensor(s_param, device=device))
        
        grad_vars = list(self.sdf_model.parameters()) + list(self.att_model1.parameters()) + [self.s]
        
        if self.num_materials == 2:
            grad_vars += list(self.att_model2.parameters())
        
        
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
            
            if "att_model1_state_dict" in ckpt:
                self.att_model1.load_state_dict(ckpt["att_model1_state_dict"])
                if self.num_materials == 2 and "att_model2_state_dict" in ckpt:
                    self.att_model2.load_state_dict(ckpt["att_model2_state_dict"])
            elif "att_model_state_dict" in ckpt:
                self.att_model1.load_state_dict(ckpt["att_model_state_dict"])
            
            self.s.data = ckpt["s"]

        self.writer = SummaryWriter(self.expdir)
        self.writer.add_text("parameters", self.args2string(cfg), global_step=0)
        
        if self.use_wandb:
            wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                name=cfg["exp"]["expname"],
                config=cfg,
                dir=self.expdir
            )
            wandb.watch([self.sdf_model, self.att_model1], log="all", log_freq=100)
        
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
        
        # Coarse-to-fine frequency regularization (paper Eq. 8-9, works for both freq and hash)
        tau = None
        if self.use_freq_reg:
            if global_step < self.warmup_iters:
                tau = None
            else:
                iters_after_warmup = global_step - self.warmup_iters
                total_iters = self.epochs * len(self.train_dloader)
                half_iters = total_iters // 2
                effective_half_iters = max(half_iters - self.warmup_iters, 1)
                
                if self.encoding_type == 'freq':
                    max_freq_L = self.sdf_model.encoder.N_freqs
                else:  # hash encoding: regularize over resolution levels
                    max_freq_L = self.sdf_model.num_levels
                
                if iters_after_warmup < effective_half_iters:
                    progress = iters_after_warmup / effective_half_iters
                    tau = self.tau_start + progress * (max_freq_L - self.tau_start)
                else:
                    tau = float(max_freq_L)
        
        # Stratified sampling (paper Section III-A2)
        t_vals = torch.linspace(0., 1., steps=self.n_samples, device=self.device)
        z_vals = near * (1. - t_vals) + far * t_vals  # [batch, n_rays, n_samples]
        
        if self.perturb:
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
            lower = torch.cat([z_vals[..., :1], mids], dim=-1)
            t_rand = torch.rand(z_vals.shape, device=self.device)
            z_vals = lower + (upper - lower) * t_rand
        
        sampled_points = ray_origins.unsqueeze(2) + ray_directions.unsqueeze(2) * z_vals.unsqueeze(-1)
        sampled_points_flat = sampled_points.reshape(-1, 3)
        
        # Forward pass
        if self.num_materials == 1:
            # 1M-NeAS: single SDF and single attenuation network
            sdf_distances, feature_vector = self.sdf_model(sampled_points_flat, tau=tau)
            boundary_values = surface_boundary_function(sdf_distances, self.s)
            attenuation_values = self.att_model1(feature_vector)
            
            # Attenuation coefficient
            att_coeff = attenuation_values.squeeze(-1) * boundary_values
        else:
            # 2M-NeAS: dual SDFs and dual attenuation networks
            d1, d2, feature_vector = self.sdf_model(sampled_points_flat, tau=tau)
            
            # Compute boundary values for both materials
            boundary_values1 = surface_boundary_function(d1, self.s)
            boundary_values2 = surface_boundary_function(d2, self.s)
            
            # Compute attenuation from both networks (both use same feature vector)
            attenuation_values1 = self.att_model1(feature_vector)
            attenuation_values2 = self.att_model2(feature_vector)
            
            # Apply boundary functions
            mu1 = attenuation_values1.squeeze(-1) * boundary_values1
            mu2 = attenuation_values2.squeeze(-1) * boundary_values2
            
            # Use selector function to choose between mu1 and mu2 based on d2
            att_coeff = selector_function(d2, mu1, mu2)
        
        att_coeff = att_coeff.reshape(batch_size, n_rays, self.n_samples)
        
        dists = z_vals[..., 1:] - z_vals[..., :-1]  # [batch, n_rays, n_samples-1]
        dists = torch.cat([dists, torch.ones_like(dists[..., :1]) * 1e-10], dim=-1)  # [batch, n_rays, n_samples]
        dists = dists * torch.norm(ray_directions, dim=-1, keepdim=True)  # scale by ray direction norm
        
        pred_intensity = volume_render_intensity(att_coeff, dists)
        gt_intensity = torch.exp(-projs)
        
        # Intensity loss
        L_int = torch.nn.functional.mse_loss(pred_intensity, gt_intensity)
        
        # Eikonal regularization
        pts_eikonal = sampled_points_flat.clone().detach().requires_grad_(True)
        
        if self.num_materials == 1:
            sdf_eikonal, _ = self.sdf_model(pts_eikonal, tau=tau)
            sdf_sum = sdf_eikonal.sum()
            
            n = torch.autograd.grad(
                outputs=sdf_sum,
                inputs=pts_eikonal,
                create_graph=True,
            )[0]
            
            n_norm = torch.linalg.norm(n, dim=-1)
            L_reg = torch.mean((n_norm - 1.0)**2)
        else:
            d1_eikonal, d2_eikonal, _ = self.sdf_model(pts_eikonal, tau=tau)
            
            n1 = torch.autograd.grad(
                outputs=d1_eikonal.sum(),
                inputs=pts_eikonal,
                create_graph=True,
                retain_graph=True
            )[0]
            n1_norm = torch.linalg.norm(n1, dim=-1)
            L_reg1 = torch.mean((n1_norm - 1.0)**2)
            
            n2 = torch.autograd.grad(
                outputs=d2_eikonal.sum(),
                inputs=pts_eikonal,
                create_graph=True,
            )[0]
            n2_norm = torch.linalg.norm(n2, dim=-1)
            L_reg2 = torch.mean((n2_norm - 1.0)**2)
            
            L_reg = (L_reg1 + L_reg2) / 2.0
        
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
        
        if self.use_wandb:
            wandb.log({
                "train/loss": L_total.item(),
                "train/loss_int": L_int.item(),
                "train/loss_reg": L_reg.item(),
                "train/epoch": global_step / len(self.train_dloader)
            }, step=global_step)
        
        return loss

    def sample_3d_volume(self, chunk_size=8192):
        """Sample a 3D volume from the learned SDF.
        
        This queries the network at each voxel position to reconstruct
        the attenuation coefficient volume for comparison with ground truth.
        
        Args:
            chunk_size: Number of voxels to process at once (to avoid OOM)
            
        Returns:
            pred_volume: Predicted attenuation volume [n1, n2, n3]
            gt_volume: Ground truth attenuation volume [n1, n2, n3]
        """
        print("Sampling 3D volume from SDF...")
        
        gt_volume = self.eval_dset.image.cpu().numpy()  # Shape: (n1, n2, n3)
        voxels = self.eval_dset.voxels  # Shape: (n1, n2, n3, 3)
        
        n1, n2, n3, _ = voxels.shape
        total_voxels = n1 * n2 * n3
        
        voxels_flat = voxels.reshape(-1, 3)  # Shape: (n1*n2*n3, 3)
        voxels_flat = torch.tensor(voxels_flat, dtype=torch.float32, device=self.device)
        
        # Process in chunks to avoid OOM
        pred_attenuation = []
        
        with torch.no_grad():
            for i in range(0, total_voxels, chunk_size):
                chunk_voxels = voxels_flat[i:i+chunk_size]
                
                if self.num_materials == 1:
                    # 1M-NeAS: single SDF and single attenuation network
                    sdf_distances, feature_vector = self.sdf_model(chunk_voxels, tau=None)
                    boundary_values = surface_boundary_function(sdf_distances, self.s)
                    attenuation_values = self.att_model1(feature_vector)
                    
                    # Attenuation coefficient = mu * boundary_function(d)
                    att_coeff = attenuation_values.squeeze(-1) * boundary_values
                    
                else:
                    # 2M-NeAS: dual SDFs and dual attenuation networks
                    d1, d2, feature_vector = self.sdf_model(chunk_voxels, tau=None)
                    
                    # Compute boundary values for both materials
                    boundary_values1 = surface_boundary_function(d1, self.s)
                    boundary_values2 = surface_boundary_function(d2, self.s)
                    
                    # Compute attenuation from both networks
                    attenuation_values1 = self.att_model1(feature_vector)
                    attenuation_values2 = self.att_model2(feature_vector)
                    
                    # Apply boundary functions
                    mu1 = attenuation_values1.squeeze(-1) * boundary_values1
                    mu2 = attenuation_values2.squeeze(-1) * boundary_values2
                    
                    # Use selector function to choose between mu1 and mu2 based on d2
                    from .network import selector_function
                    att_coeff = selector_function(d2, mu1, mu2)
                
                pred_attenuation.append(att_coeff.cpu())
                
        # Concatenate all chunks and reshape to 3D volume
        pred_attenuation = torch.cat(pred_attenuation, dim=0)
        pred_volume = pred_attenuation.reshape(n1, n2, n3).numpy()
        
        print(f"Volume sampled: shape={pred_volume.shape}, "
              f"pred range=[{pred_volume.min():.6f}, {pred_volume.max():.6f}], "
              f"gt range=[{gt_volume.min():.6f}, {gt_volume.max():.6f}]")
        
        return pred_volume, gt_volume

    def eval_step(self, global_step, idx_epoch):
        """Evaluation step with projection and 3D volume metrics."""
        print("Running validation...")
        torch.cuda.empty_cache()
        
        self.sdf_model.eval()
        self.att_model1.eval()
        if self.num_materials == 2:
            self.att_model2.eval()
        
        val_save_dir = osp.join(self.expdir, f'val_epoch_{idx_epoch}')
        os.makedirs(val_save_dir, exist_ok=True)

        with torch.no_grad():
            select_ind = np.random.choice(len(self.eval_dset))
            rays = self.eval_dset.rays[select_ind].to(self.device)
            projs = self.eval_dset.projs[select_ind].to(self.device)
            
            img = render_image(rays, self.sdf_model, self.att_model1, self.s, 
                             self.val_n_samples, chunk_size=self.val_chunk_size, tau=None, 
                             att_model2=self.att_model2 if self.num_materials == 2 else None)

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
            
            proj_mse = get_mse(img, torch.exp(-projs)).item()
            proj_psnr = get_psnr(img, torch.exp(-projs)).item()
            proj_ssim = get_ssim(img, torch.exp(-projs)).item()
            
            # Sample a 3D volume from the SDF
            pred_volume, gt_volume = self.sample_3d_volume(chunk_size=8192)
            
            vol_psnr_3d = get_psnr_3d(pred_volume, gt_volume)
            vol_ssim_3d = get_ssim_3d(pred_volume, gt_volume)
            
            self.writer.add_scalar("eval/proj_mse", proj_mse, global_step)
            self.writer.add_scalar("eval/proj_psnr", proj_psnr, global_step)
            self.writer.add_scalar("eval/proj_ssim", proj_ssim, global_step)
            self.writer.add_scalar("eval/3d_psnr", vol_psnr_3d, global_step)
            self.writer.add_scalar("eval/3d_ssim", vol_ssim_3d, global_step)
            self.writer.add_image("eval/proj_pred", cast_to_image(img.cpu().numpy().T), global_step, dataformats="HWC")
            
            if self.use_wandb:
                wandb.log({
                    "eval/proj_mse": proj_mse,
                    "eval/proj_psnr": proj_psnr,
                    "eval/proj_ssim": proj_ssim,
                    "eval/3d_psnr": vol_psnr_3d,
                    "eval/3d_ssim": vol_ssim_3d,
                    "eval/epoch": idx_epoch,
                    "eval/proj_pred": wandb.Image(img.cpu().numpy().T, caption=f"Prediction - Epoch {idx_epoch}"),
                    "eval/proj_gt": wandb.Image(torch.exp(-projs).cpu().numpy().T, caption=f"Ground Truth - Epoch {idx_epoch}")
                }, step=global_step)
                
            print(f"Eval metrics - Proj MSE: {proj_mse:.6f}, Proj PSNR: {proj_psnr:.2f}, Proj SSIM: {proj_ssim:.4f}, 3D PSNR: {vol_psnr_3d:.2f}, 3D SSIM: {vol_ssim_3d:.4f}")
            
        self.sdf_model.train()
        self.att_model1.train()
        if self.num_materials == 2:
            self.att_model2.train()

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
                    'att_model1_state_dict': self.att_model1.state_dict(),
                    's': self.s.data,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.lr_scheduler.state_dict(),
                    'loss_history': self.loss_history,
                    'feature_dim': self.conf["network"].get("feature_dim", 8),
                    'encoding_type': self.encoding_type,
                    'num_materials': self.num_materials,
                    'alpha1': self.conf["network"].get("alpha1", self.conf["network"].get("alpha", 3.4)),
                    'beta1': self.conf["network"].get("beta1", self.conf["network"].get("beta", 0.1)),
                }
                
                if self.num_materials == 2:
                    checkpoint_dict['att_model2_state_dict'] = self.att_model2.state_dict()
                    checkpoint_dict['alpha2'] = self.conf["network"].get("alpha2", 5.5)
                    checkpoint_dict['beta2'] = self.conf["network"].get("beta2", 3.5)
                
                if self.encoding_type == 'hash':
                    checkpoint_dict['num_levels'] = self.conf["network"].get("num_levels", 14)
                    checkpoint_dict['level_dim'] = self.conf["network"].get("level_dim", 2)
                    checkpoint_dict['base_resolution'] = self.conf["network"].get("base_resolution", 16)
                    checkpoint_dict['log2_hashmap_size'] = self.conf["network"].get("log2_hashmap_size", 19)
                
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
        
        if self.use_wandb:
            wandb.finish()
