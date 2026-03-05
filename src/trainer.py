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
from .network import (sdf_freq_mlp, att_freq_mlp, sdf_hash_mlp, att_hash_mlp,
                      sdf_freq_mlp_km, sdf_hash_mlp_km,
                      shared_att_freq_mlp, shared_att_hash_mlp,
                      nested_material_selector)
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
        self.lambda_mask = cfg["train"].get("lambda_mask", 0.0)
        self.n_mask_rays = cfg["train"].get("n_mask_rays", 0) if self.lambda_mask > 0 else 0
        self.device = device
        
        # Frequency regularization (coarse-to-fine)
        self.use_freq_reg = cfg["train"].get("use_freq_reg", False)
        self.tau_start = cfg["train"].get("tau_start", 2.0)
        self.warmup_iters = cfg["train"].get("warmup_iters", 500)
        
        self.use_wandb = cfg["log"].get("use_wandb", True)
        self.wandb_project = cfg["log"].get("wandb_project", "neas_experimental")
        self.wandb_entity = cfg["log"].get("wandb_entity", None)
  
        # Log directory
        self.expdir = osp.join(cfg["exp"]["expdir"], cfg["exp"]["expname"])
        self.ckptdir = osp.join(self.expdir, "checkpoint.pth")
        self.ckptdir_backup = osp.join(self.expdir, "checkpoint_backup.pth")
        self.evaldir = osp.join(self.expdir, "eval")
        os.makedirs(self.expdir, exist_ok=True)
        os.makedirs(self.evaldir, exist_ok=True)

        # Dataset
        num_views = cfg["train"].get("num_views", None)
        train_dset = TIGREDataset(cfg["exp"]["datadir"], cfg["train"]["n_rays"], "train", device, num_views=num_views, n_mask_rays=self.n_mask_rays)
        self.eval_dset = TIGREDataset(cfg["exp"]["datadir"], cfg["train"]["n_rays"], "val", device) if self.i_eval > 0 else None
        self.train_dloader = torch.utils.data.DataLoader(train_dset, batch_size=cfg["train"]["n_batch"], shuffle=True)
        
        self.geo = train_dset.geo
        self.n_train_images = train_dset.n_samples
    
        # Network
        feature_dim = cfg["network"].get("feature_dim", 8)
        multires = cfg["network"].get("multires", 6)
        encoding_type = cfg["network"].get("encoding_type", "freq")  # 'freq' or 'hash'
        num_materials = cfg["network"].get("num_materials", 1)  # 1 for 1M-NeAS, >=2 for KM-NeAS
        
        self.num_materials = num_materials
        
        # Build per-material activation configs: list of (alpha, beta)
        material_configs = [(m['alpha'], m['beta']) for m in cfg['network']['materials']]
        self.material_configs = material_configs
        
        # Create networks
        if encoding_type == "freq":
            if num_materials == 1:
                self.sdf_model = sdf_freq_mlp(input_dim=3, output_dim=1, feature_dim=feature_dim, multires=multires).to(device)
                self.att_model1 = att_freq_mlp(input_dim=feature_dim, output_dim=1,
                                              alpha=material_configs[0][0], beta=material_configs[0][1]).to(device)
                self.att_model = None
            else:  # K >= 2: shared latent space
                self.sdf_model = sdf_freq_mlp_km(input_dim=3, num_materials=num_materials,
                                                 feature_dim=feature_dim, multires=multires).to(device)
                self.att_model = shared_att_freq_mlp(input_dim=feature_dim,
                                                     material_activations=material_configs).to(device)
                self.att_model1 = None
                
        elif encoding_type == "hash":
            num_levels = cfg["network"].get("num_levels", 14)
            level_dim = cfg["network"].get("level_dim", 2)
            base_resolution = cfg["network"].get("base_resolution", 16)
            log2_hashmap_size = cfg["network"].get("log2_hashmap_size", 19)
            
            if num_materials == 1:
                self.sdf_model = sdf_hash_mlp(input_dim=3, output_dim=1, feature_dim=feature_dim,
                                             num_levels=num_levels, level_dim=level_dim,
                                             base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size).to(device)
                self.att_model1 = att_hash_mlp(input_dim=feature_dim, output_dim=1,
                                              num_levels=num_levels, level_dim=level_dim,
                                              base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size,
                                              alpha=material_configs[0][0], beta=material_configs[0][1]).to(device)
                self.att_model = None
            else:  # K >= 2: shared latent space
                self.sdf_model = sdf_hash_mlp_km(input_dim=3, num_materials=num_materials,
                                                 feature_dim=feature_dim,
                                                 num_levels=num_levels, level_dim=level_dim,
                                                 base_resolution=base_resolution,
                                                 log2_hashmap_size=log2_hashmap_size).to(device)
                self.att_model = shared_att_hash_mlp(input_dim=feature_dim,
                                                     material_activations=material_configs).to(device)
                self.att_model1 = None
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
        
        self.encoding_type = encoding_type
        
        # Boundary sharpness parameter (learnable)
        s_param = cfg["network"].get("s_param", 20.0)
        self.s = torch.nn.Parameter(torch.tensor(s_param, device=device))
        
        if self.num_materials == 1:
            grad_vars = list(self.sdf_model.parameters()) + list(self.att_model1.parameters()) + [self.s]
        else:
            grad_vars = list(self.sdf_model.parameters()) + list(self.att_model.parameters()) + [self.s]
        
        
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
            
            if self.num_materials == 1:
                if "att_model1_state_dict" in ckpt:
                    self.att_model1.load_state_dict(ckpt["att_model1_state_dict"])
                elif "att_model_state_dict" in ckpt:
                    self.att_model1.load_state_dict(ckpt["att_model_state_dict"])
            else:
                if "att_model_shared_state_dict" in ckpt:
                    self.att_model.load_state_dict(ckpt["att_model_shared_state_dict"])
            
            self.s.data = ckpt["s"]

        self.writer = SummaryWriter(self.expdir)
        self.writer.add_text("parameters", self.args2string(cfg), global_step=0)
        
        self.config_path = cfg.get("_config_path", None)
        if self.config_path is not None and osp.exists(self.config_path):
            try:
                copyfile(self.config_path, osp.join(self.expdir, osp.basename(self.config_path)))
            except Exception:
                pass
            try:
                copyfile(self.config_path, osp.join(self.expdir, "config.yaml"))
            except Exception:
                pass

        if self.use_wandb:
            wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                name=cfg["exp"]["expname"],
                config=cfg,
                dir=self.expdir
            )
            watch_models = [self.sdf_model, self.att_model1 if self.att_model1 is not None else self.att_model]
            wandb.watch(watch_models, log="all", log_freq=100)
            if self.config_path is not None and osp.exists(self.config_path):
                config_art = wandb.Artifact("training-config", type="config")
                config_art.add_file(self.config_path)
                wandb.log_artifact(config_art)
        
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
        gt_intensity = batch_data['projs_intensity'].to(self.device)

        ray_origins = rays[..., :3]
        ray_directions = rays[..., 3:6] 
        near = rays[..., 6:7]
        far = rays[..., 7:8]
        
        batch_size, n_rays, _ = ray_origins.shape
        
        # Coarse-to-fine frequency regularization (paper Eq. 8-9, works for both freq and hash)
        # Paper: τ starts at tau_start (2.0) and grows linearly to L until half of total iterations.
        # Note: warmup_iters is for pose refinement (not implemented here), not for frequency regularization.
        tau = None
        if self.use_freq_reg:
            total_iters = self.epochs * len(self.train_dloader)
            half_iters = total_iters // 2
            
            if self.encoding_type == 'freq':
                max_freq_L = self.sdf_model.encoder.N_freqs
            else:  # hash encoding: regularize over resolution levels
                max_freq_L = self.sdf_model.num_levels
            
            if half_iters > 0 and global_step < half_iters:
                progress = global_step / half_iters
                tau = self.tau_start + progress * (max_freq_L - self.tau_start)
            else:
                tau = float(max_freq_L)
        
        # Stratified sampling (paper Section III-A2)
        t_vals = torch.linspace(0., 1., steps=self.n_samples, device=self.device)
        z_vals = near * (1. - t_vals) + far * t_vals  # [batch, n_rays, n_samples]
        
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., :1], mids], dim=-1)
        t_rand = torch.rand(z_vals.shape, device=self.device)
        z_vals = lower + (upper - lower) * t_rand
        
        sampled_points = ray_origins.unsqueeze(2) + ray_directions.unsqueeze(2) * z_vals.unsqueeze(-1)
        sampled_points_flat = sampled_points.reshape(-1, 3).detach().requires_grad_(True)
        
        # Forward pass
        if self.num_materials == 1:
            # 1M-NeAS: single SDF and single attenuation network
            sdf_distances, feature_vector = self.sdf_model(sampled_points_flat, tau=tau)
            boundary_values = surface_boundary_function(sdf_distances, self.s)
            attenuation_values = self.att_model1(feature_vector)
            
            # Attenuation coefficient
            att_coeff = attenuation_values.squeeze(-1) * boundary_values
        else:
            # KM-NeAS: K SDFs + shared attenuation backbone + nested selector
            distances, feature_vector = self.sdf_model(sampled_points_flat, tau=tau)
            
            # Compute boundary values for all K materials
            bv = [surface_boundary_function(d, self.s) for d in distances]
            
            # Shared attenuation backbone → K raw attenuation heads
            raw_attenuations = self.att_model(feature_vector)
            
            # Nested K-material priority selector
            att_coeff = nested_material_selector(bv, raw_attenuations)
        
        att_coeff = att_coeff.reshape(batch_size, n_rays, self.n_samples)
        
        dists = z_vals[..., 1:] - z_vals[..., :-1]  # [batch, n_rays, n_samples-1]
        dists = torch.cat([dists, torch.ones_like(dists[..., :1]) * 1e-10], dim=-1)  # [batch, n_rays, n_samples]
        dists = dists * torch.norm(ray_directions, dim=-1, keepdim=True)  # scale by ray direction norm
        
        pred_intensity = volume_render_intensity(att_coeff, dists)
        
        # Intensity loss
        L_int = torch.nn.functional.mse_loss(pred_intensity, gt_intensity)
        
        if self.num_materials == 1:
            n = torch.autograd.grad(
                outputs=sdf_distances.sum(),
                inputs=sampled_points_flat,
                create_graph=True,
            )[0]
            
            n_norm = torch.linalg.norm(n, dim=-1)
            L_reg = torch.mean((n_norm - 1.0)**2)
        else:
            # Eikonal regularization for K SDF fields
            L_reg = torch.tensor(0.0, device=self.device)
            K = len(distances)
            for i in range(K):
                n_i = torch.autograd.grad(
                    outputs=distances[i].sum(),
                    inputs=sampled_points_flat,
                    create_graph=True,
                    retain_graph=True
                )[0]
                n_i_norm = torch.linalg.norm(n_i, dim=-1)
                L_reg = L_reg + torch.mean((n_i_norm - 1.0)**2)
            L_reg = L_reg / K
        
        # Total loss
        L_total = L_int + self.lambda_reg * L_reg
        
        # Mask loss: penalize non-zero attenuation predictions in air regions.
        L_mask = torch.tensor(0.0, device=self.device)
        if self.lambda_mask > 0 and 'mask_rays' in batch_data:
            current_epoch = global_step // len(self.train_dloader)
            max_mask_epoch = int(self.epochs * 0.2)
            if current_epoch < max_mask_epoch:
                mask_rays = batch_data['mask_rays'].to(self.device)          # [B, n_mask, 8]
                mask_gt = batch_data['mask_projs_intensity'].to(self.device) # [B, n_mask]

                mask_ray_o = mask_rays[..., :3]
                mask_ray_d = mask_rays[..., 3:6]
                mask_near  = mask_rays[..., 6:7]
                mask_far   = mask_rays[..., 7:8]

                B_m, N_m, _ = mask_ray_o.shape

                t_m = torch.linspace(0., 1., steps=self.n_samples, device=self.device)
                z_m = mask_near * (1. - t_m) + mask_far * t_m
                mids_m = .5 * (z_m[..., 1:] + z_m[..., :-1])
                z_m = torch.cat([z_m[..., :1], mids_m], dim=-1) + \
                      (torch.cat([mids_m, z_m[..., -1:]], dim=-1) -
                       torch.cat([z_m[..., :1], mids_m], dim=-1)) * torch.rand(z_m.shape, device=self.device)

                pts_m = mask_ray_o.unsqueeze(2) + mask_ray_d.unsqueeze(2) * z_m.unsqueeze(-1)
                pts_m_flat = pts_m.reshape(-1, 3)

                if self.num_materials == 1:
                    sdf_m, feat_m = self.sdf_model(pts_m_flat, tau=tau)
                    bv_m = surface_boundary_function(sdf_m, self.s)
                    att_m = self.att_model1(feat_m).squeeze(-1) * bv_m
                else:
                    dists_m, feat_m = self.sdf_model(pts_m_flat, tau=tau)
                    bv_m = [surface_boundary_function(d, self.s) for d in dists_m]
                    raw_att_m = self.att_model(feat_m)
                    att_m = nested_material_selector(bv_m, raw_att_m)

                att_m = att_m.reshape(B_m, N_m, self.n_samples)
                dists_m = z_m[..., 1:] - z_m[..., :-1]
                dists_m = torch.cat([dists_m, torch.ones_like(dists_m[..., :1]) * 1e-10], dim=-1)
                dists_m = dists_m * torch.norm(mask_ray_d, dim=-1, keepdim=True)

                pred_mask = volume_render_intensity(att_m, dists_m)
                L_mask = torch.nn.functional.mse_loss(pred_mask, mask_gt)
                L_total = L_total + self.lambda_mask * L_mask
            else:
                pass
        
        loss = {
            "loss": L_total,
            "loss_int": L_int,
            "loss_reg": L_reg,
            "loss_mask": L_mask,
        }
        
        # Log
        self.writer.add_scalar("train/loss", L_total.item(), global_step)
        self.writer.add_scalar("train/loss_int", L_int.item(), global_step)
        self.writer.add_scalar("train/loss_reg", L_reg.item(), global_step)
        self.writer.add_scalar("train/loss_mask", L_mask.item(), global_step)
        
        if self.use_wandb:
            wandb.log({
                "train/loss": L_total.item(),
                "train/loss_int": L_int.item(),
                "train/loss_reg": L_reg.item(),
                "train/loss_mask": L_mask.item(),
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
        
        gt_volume = self.eval_dset.image.cpu()  # Shape: (n1, n2, n3) — kept as a tensor
        voxels = self.eval_dset.voxels  # Shape: (n1, n2, n3, 3)
        
        n1, n2, n3, _ = voxels.shape
        total_voxels = n1 * n2 * n3
        
        voxels_flat = voxels.reshape(-1, 3).to(device=self.device, dtype=torch.float32)
        
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
                    # KM-NeAS: K SDFs + shared attenuation + nested selector
                    distances, feature_vector = self.sdf_model(chunk_voxels, tau=None)
                    bv = [surface_boundary_function(d, self.s) for d in distances]
                    raw_attenuations = self.att_model(feature_vector)
                    att_coeff = nested_material_selector(bv, raw_attenuations)
                
                pred_attenuation.append(att_coeff.cpu())
                
        # Concatenate all chunks and reshape to 3D volume
        pred_attenuation = torch.cat(pred_attenuation, dim=0)
        pred_volume = pred_attenuation.reshape(n1, n2, n3)
        
        dot_pg = (pred_volume * gt_volume).sum()
        dot_pp = (pred_volume * pred_volume).sum()
        if dot_pp > 0:
            scale_factor = dot_pg / dot_pp
            pred_volume = pred_volume * scale_factor
            print(f"Applied normalization scale factor: {scale_factor:.4f}")
        
        print(f"Volume sampled: shape={tuple(pred_volume.shape)}, "
              f"pred range=[{pred_volume.min():.6f}, {pred_volume.max():.6f}], "
              f"gt range=[{gt_volume.min():.6f}, {gt_volume.max():.6f}]")
        
        return pred_volume, gt_volume

    def eval_step(self, global_step, idx_epoch):
        """Evaluation step with projection and 3D volume metrics."""
        print("Running validation...")
        torch.cuda.empty_cache()
        
        self.sdf_model.eval()
        if self.num_materials == 1:
            self.att_model1.eval()
        else:
            self.att_model.eval()
        
        val_save_dir = osp.join(self.expdir, f'val_epoch_{idx_epoch}')
        os.makedirs(val_save_dir, exist_ok=True)

        with torch.no_grad():
            select_ind = np.random.choice(len(self.eval_dset))
            rays = self.eval_dset.rays[select_ind].to(self.device)
            projs_gt = self.eval_dset.projs_intensity[select_ind].to(self.device)
            
            att_for_render = self.att_model1 if self.num_materials == 1 else self.att_model
            img = render_image(rays, self.sdf_model, att_for_render, self.s, 
                             self.val_n_samples, chunk_size=self.val_chunk_size, tau=None, 
                             num_materials=self.num_materials)

            # Visualization
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(img.cpu().numpy().T, cmap='gray')
            plt.title('Predicted')
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(projs_gt.cpu().numpy().T, cmap='gray')
            plt.title('Ground Truth')
            plt.axis('off')
            plt.savefig(os.path.join(val_save_dir, f'val_{select_ind}.png'))
            plt.close()
            
            projs_pred = img

            proj_mse = get_mse(projs_pred, projs_gt).item()
            proj_psnr = get_psnr(projs_pred, projs_gt).item()
            proj_ssim = get_ssim(projs_pred, projs_gt).item()
            
            # Sample a 3D volume from the SDF
            image_pred, image = self.sample_3d_volume(chunk_size=8192)
            
            vol_psnr_3d = get_psnr_3d(image_pred, image)
            vol_ssim_3d = get_ssim_3d(image_pred, image)

            projs = projs_gt

            loss = {
                "proj_mse": proj_mse,
                "proj_psnr": proj_psnr,
                "psnr_3d": vol_psnr_3d,
                "ssim_3d": vol_ssim_3d,
            }

            print(f"Eval metrics - Proj MSE: {proj_mse:.6f}, Proj PSNR: {proj_psnr:.2f}, "
                  f"Proj SSIM: {proj_ssim:.4f}, 3D PSNR: {vol_psnr_3d:.2f}, 3D SSIM: {vol_ssim_3d:.4f}")

            # Logging
            show_slice = 5
            show_step = image.shape[-1]//show_slice
            show_image = image[...,::show_step]
            show_image_pred = image_pred[...,::show_step]
            show = []
            for i_show in range(show_slice):
                show.append(torch.concat([show_image[..., i_show], show_image_pred[..., i_show]], dim=0))
            show_density = torch.concat(show, dim=1)
            show_proj = torch.concat([projs_gt, projs_pred], dim=1)

            self.writer.add_image("eval/density (row1: gt, row2: pred)", cast_to_image(show_density), global_step, dataformats="HWC")
            self.writer.add_image("eval/projection (left: gt, right: pred)", cast_to_image(show_proj), global_step, dataformats="HWC")

            for ls in loss.keys():
                self.writer.add_scalar(f"eval/{ls}", loss[ls], global_step)
            if self.use_wandb:
                wandb.log({
                    "eval/proj_mse": loss["proj_mse"],
                    "eval/proj_psnr": loss["proj_psnr"],
                    "eval/psnr_3d": loss["psnr_3d"],
                    "eval/ssim_3d": loss["ssim_3d"],
                    "eval/density_comparison": wandb.Image(cast_to_image(show_density), caption="Top: GT, Bottom: Pred"),
                    "eval/projection_comparison": wandb.Image(cast_to_image(show_proj), caption="Left: GT, Right: Pred")
                }, step=self.global_step)
            
        self.sdf_model.train()
        if self.num_materials == 1:
            self.att_model1.train()
        else:
            self.att_model.train()

    def start(self):
        """Main training loop."""
        iter_per_epoch = len(self.train_dloader)
        pbar = tqdm(total=iter_per_epoch * self.epochs, leave=True)
        if self.epoch_start > 0:
            pbar.update(self.epoch_start * iter_per_epoch)

        for idx_epoch in range(self.epoch_start, self.epochs + 1):
            
            # Evaluate
            if (idx_epoch % self.i_eval == 0 or idx_epoch == self.epochs) and self.i_eval > 0:
                self.eval_step(global_step=self.global_step, idx_epoch=idx_epoch)
            
            # Save checkpoint
            if (idx_epoch % self.i_save == 0 or idx_epoch == self.epochs) and idx_epoch > 0:
                save_path = osp.join(self.expdir, f'checkpoint_epoch_{idx_epoch}.pth')
                checkpoint_dict = {
                    'epoch': idx_epoch,
                    'args': self.conf,
                    'sdf_model_state_dict': self.sdf_model.state_dict(),
                    's': self.s.data,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.lr_scheduler.state_dict(),
                    'loss_history': self.loss_history,
                    'feature_dim': self.conf["network"].get("feature_dim", 8),
                    'encoding_type': self.encoding_type,
                    'num_materials': self.num_materials,
                    'material_configs': self.material_configs,
                }
                
                if self.num_materials == 1:
                    checkpoint_dict['att_model1_state_dict'] = self.att_model1.state_dict()
                else:
                    checkpoint_dict['att_model_shared_state_dict'] = self.att_model.state_dict()
                
                if self.encoding_type == 'hash':
                    checkpoint_dict['num_levels'] = self.conf["network"].get("num_levels", 14)
                    checkpoint_dict['level_dim'] = self.conf["network"].get("level_dim", 2)
                    checkpoint_dict['base_resolution'] = self.conf["network"].get("base_resolution", 16)
                    checkpoint_dict['log2_hashmap_size'] = self.conf["network"].get("log2_hashmap_size", 19)
                
                torch.save(checkpoint_dict, save_path)
                try:
                    torch.save(checkpoint_dict, self.ckptdir)
                except Exception:
                    pass
                print(f"Checkpoint saved at epoch {idx_epoch}")
                try:
                    torch.save(checkpoint_dict, self.ckptdir_backup)
                except Exception:
                    pass
                if self.use_wandb:
                    art_name = f"model_epoch_{idx_epoch}"
                    model_art = wandb.Artifact(art_name, type="model")
                    model_art.add_file(save_path)
                    if self.config_path is not None and osp.exists(self.config_path):
                        model_art.add_file(self.config_path)
                    wandb.log_artifact(model_art)
            
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
            
            if idx_epoch % 10 == 0:
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
