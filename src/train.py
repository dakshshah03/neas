from mlp import create_neas_model
from dataset import TIGREDataset
from torch.utils.data import DataLoader
import torch
import os
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Train NeAS model")
    
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--n_samples", type=int, default=128, help="Number of samples per ray during training")
    parser.add_argument("--lambda_reg", type=float, default=0.01, help="Regularization strength")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=512, help="Number of rays per image")
    parser.add_argument("--lr_step_size", type=int, default=50, help="StepLR step size (epochs)")
    parser.add_argument("--lr_gamma", type=float, default=0.5, help="StepLR gamma")
    
    parser.add_argument("--data_path", type=str, default="data/foot_50.pickle", help="Path to data file")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/", help="Directory to save checkpoints")
    parser.add_argument("--save_interval", type=int, default=100, help="Epoch interval for saving checkpoints")
    
    parser.add_argument("--feature_dim", type=int, default=8, help="Feature dimension for SDF model")
    parser.add_argument("--s_param", type=float, default=20.0, help="Initial value for s (boundary sharpness) parameter")
    parser.add_argument("--val_chunk_size", type=int, default=4096, help="Chunk size for validation rendering")
    parser.add_argument("--val_n_samples", type=int, default=128, help="Number of samples per ray during validation (higher for better quality)")
    
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha parameter for attenuation activation function")
    parser.add_argument("--beta", type=float, default=0.0, help="Beta parameter for attenuation activation function")
    
    parser.add_argument("--encoding", type=str, default="frequency", choices=["frequency", "hashgrid"], help="Encoding type: frequency or hashgrid")
    parser.add_argument("--multires", type=int, default=6, help="Number of frequency bands for frequency encoding")
    parser.add_argument("--num_levels", type=int, default=14, help="Number of levels for hash encoding")
    parser.add_argument("--level_dim", type=int, default=2, help="Feature dimension per level for hash encoding")
    parser.add_argument("--base_resolution", type=int, default=16, help="Base resolution for hash encoding")
    parser.add_argument("--log2_hashmap_size", type=int, default=19, help="Log2 of hash table size")
    
    parser.add_argument("--use_freq_reg", action="store_true", help="Enable frequency regularization (coarse-to-fine)")
    parser.add_argument("--tau_start", type=float, default=2.0, help="Initial tau value for frequency regularization")
    parser.add_argument("--warmup_iters", type=int, default=500, help="Number of warmup iterations before applying frequency regularization")
    
    return parser.parse_args()

def surface_boundary_function(d, s):
    return torch.exp(-s*d)/(1 + torch.exp(-s*d))

def volume_render_intensity(att_coeff, dists):
    mu_delta_sum = torch.sum(att_coeff * dists, dim=-1) # [batch_size, n_rays]
    I_hat = torch.exp(-mu_delta_sum) # [batch_size, n_rays]
    return I_hat

def render_image(rays, sdf_model, att_model, s, n_samples, chunk_size=4096, tau=None):
    device = rays.device
    H, W, _ = rays.shape
    rays_flat = rays.reshape(-1, 8)
    
    pred_intensities = []
    
    with torch.no_grad():
        for i in range(0, rays_flat.shape[0], chunk_size):
            chunk_rays = rays_flat[i:i+chunk_size]
            
            ray_origins = chunk_rays[..., :3]
            ray_directions = chunk_rays[..., 3:6] 
            near = chunk_rays[..., 6:7]
            far = chunk_rays[..., 7:8]
            
            # ray point sampling
            t_vals = torch.linspace(0., 1., steps=n_samples, device=device)
            sample_depths = near * (1. - t_vals) + far * t_vals
            sampled_points = ray_origins.unsqueeze(1) + ray_directions.unsqueeze(1) * sample_depths.unsqueeze(-1)
            sampled_points_flat = sampled_points.reshape(-1, 3)
            
            sdf_distances, feature_vector = sdf_model(sampled_points_flat, tau=tau)
            boundary_values = surface_boundary_function(sdf_distances, s)
            attenuation_values = att_model(feature_vector)
            
            att_coeff = attenuation_values.squeeze(-1) * boundary_values
            att_coeff = att_coeff.reshape(chunk_rays.shape[0], n_samples)
            
            dists = (far - near) / n_samples
            chunk_intensity = volume_render_intensity(att_coeff, dists)
            pred_intensities.append(chunk_intensity)
            
    pred_intensity = torch.cat(pred_intensities, dim=0)
    return pred_intensity.reshape(H, W)

def train(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, f"{dataset_name}_{timestamp}")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {args.checkpoint_dir}")
    train_data = TIGREDataset(path=args.data_path, n_rays=args.batch_size, type="train")
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=0)

    val_data = TIGREDataset(path=args.data_path, type="val")
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Using encoding: {args.encoding}")

    # Create models using the unified interface
    sdf_model, att_model = create_neas_model(
        encoding=args.encoding,
        feature_dim=args.feature_dim,
        alpha=args.alpha,
        beta=args.beta,
        multires=args.multires,
        num_levels=args.num_levels,
        level_dim=args.level_dim,
        base_resolution=args.base_resolution,
        log2_hashmap_size=args.log2_hashmap_size
    )
    sdf_model = sdf_model.to(device)
    att_model = att_model.to(device)

    s = torch.tensor(args.s_param, requires_grad=True, device=device)

    optimizer = torch.optim.Adam(
        list(sdf_model.parameters()) + list(att_model.parameters()), 
        lr=args.lr
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    loss_history = {'total': [], 'int': [], 'reg': []}
    print("Starting training...")
    
    # frequency encoding/coarse-to-fine stuff
    total_iters = args.epochs * len(train_loader)
    half_iters = total_iters // 2
    
    # N_freqs is for frequency encoding
    if args.encoding == 'frequency':
        max_freq_L = sdf_model.encoder.N_freqs
    else:
        max_freq_L = None
    
    global_iter = 0
    
    epoch_tqdm = tqdm(range(args.epochs), desc="Training")
    for epoch in epoch_tqdm:
        epoch_loss = 0.0
        epoch_int_loss = 0.0
        epoch_reg_loss = 0.0
        
        for batch_idx, batch_data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)):
            rays = batch_data['rays'].to(device)
            projs = batch_data['projs'].to(device)
            
            optimizer.zero_grad()
            
            ray_origins = rays[..., :3]
            ray_directions = rays[..., 3:6] 
            near = rays[..., 6:7]
            far = rays[..., 7:8]
            
            batch_size, n_rays, _ = ray_origins.shape
            
            # coarse-to-fine freq reg (only for frequency encoding)
            tau = None
            if args.encoding == 'frequency' and args.use_freq_reg:
                if global_iter < args.warmup_iters:
                    tau = None
                else:
                    iters_after_warmup = global_iter - args.warmup_iters
                    effective_half_iters = half_iters - args.warmup_iters
                    
                    if iters_after_warmup < effective_half_iters:
                        progress = iters_after_warmup / effective_half_iters
                        tau = args.tau_start + progress * (max_freq_L - args.tau_start)
                    else:
                        tau = float(max_freq_L)
            
            global_iter += 1
            
            # ray point sampling stuff
            t_vals = torch.linspace(0., 1., steps=args.n_samples, device=device)
            sample_depths = near * (1. - t_vals) + far * t_vals
            sampled_points = ray_origins.unsqueeze(2) + ray_directions.unsqueeze(2) * sample_depths.unsqueeze(-1)
            sampled_points_flat = sampled_points.reshape(-1, 3)
            
            # Forward pass
            sdf_distances, feature_vector = sdf_model(sampled_points_flat, tau=tau)
            boundary_values = surface_boundary_function(sdf_distances, s)
            attenuation_values = att_model(feature_vector)
            
            # attenuation coefficient
            att_coeff = attenuation_values.squeeze(-1) * boundary_values
            att_coeff = att_coeff.reshape(batch_size, n_rays, args.n_samples)
            
            # Volume rendering (equation 5)
            dists = (far - near) / args.n_samples
            pred_intensity = volume_render_intensity(att_coeff, dists)
            gt_intensity = torch.exp(-projs)
            
            # eikonal reg
            pts_eikonal = sampled_points_flat.clone().detach().requires_grad_(True)
            sdf_eikonal, _ = sdf_model(pts_eikonal, tau=tau)
            sdf_sum = sdf_eikonal.sum()
            
            # calculates the gradient of the SDF field 
            n = torch.autograd.grad(
                outputs=sdf_sum,
                inputs=pts_eikonal,
                create_graph=True,
            )[0]
            
            # loss (intensity and regularization)
            L_int = torch.nn.functional.mse_loss(pred_intensity, gt_intensity)
            n_norm = torch.linalg.norm(n, dim=-1)
            L_reg = torch.mean((n_norm - 1.0)**2)
            L_total = L_int + args.lambda_reg * L_reg
            
            L_total.backward()
            
            optimizer.step()
            
            epoch_int_loss += L_int.item()
            epoch_reg_loss += L_reg.item()
            epoch_loss += L_total.item()
        
        avg_loss = epoch_loss / len(train_loader)
        avg_int_loss = epoch_int_loss / len(train_loader)
        avg_reg_loss = epoch_reg_loss / len(train_loader)
        
        loss_history['total'].append(avg_loss)
        loss_history['int'].append(avg_int_loss)
        loss_history['reg'].append(args.lambda_reg * avg_reg_loss)

        epoch_tqdm.set_postfix({
            'loss': f"{avg_loss:.4f}",
            'int_loss': f"{avg_int_loss:.4f}",
            'reg_loss': f"{avg_reg_loss:.4f}"
        })

        plt.figure(figsize=(10, 5))
        plt.plot(loss_history['total'], label='Total Loss')
        plt.plot(loss_history['int'], label='Intensity Loss')
        plt.plot(loss_history['reg'], label='Regularization Loss (Lambda scaled)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(args.checkpoint_dir, 'loss_curve.png'))
        plt.close()
        
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'args': vars(args),
                'sdf_model_state_dict': sdf_model.state_dict(),
                'att_model_state_dict': att_model.state_dict(),
                's': s,
                'encoding': args.encoding,
                'feature_dim': args.feature_dim,
                'alpha': args.alpha,
                'beta': args.beta,
                'multires': args.multires,
                'num_levels': args.num_levels,
                'level_dim': args.level_dim,
                'base_resolution': args.base_resolution,
                'log2_hashmap_size': args.log2_hashmap_size,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'loss_history': loss_history,
            }, save_path)
            print(f"Checkpoint saved at epoch {epoch+1}")

            print("Running validation...")
            sdf_model.eval()
            att_model.eval()
            val_save_dir = os.path.join(args.checkpoint_dir, f'val_epoch_{epoch+1}')
            os.makedirs(val_save_dir, exist_ok=True)

            with torch.no_grad():
                for i, batch in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
                    rays = batch['rays'].squeeze(0).to(device) # [W, H, 8]
                    projs = batch['projs'].squeeze(0).to(device) # [W, H]
                    img = render_image(rays, sdf_model, att_model, s, args.val_n_samples, chunk_size=args.val_chunk_size, tau=None)

                    plt.figure(figsize=(10, 5))
                    plt.subplot(1, 2, 1)
                    plt.imshow(img.cpu().numpy().T, cmap='gray')
                    plt.title('Predicted')
                    plt.axis('off')
                    plt.subplot(1, 2, 2)
                    plt.imshow(torch.exp(-projs).cpu().numpy().T, cmap='gray')
                    plt.title('Ground Truth')
                    plt.axis('off')
                    plt.savefig(os.path.join(val_save_dir, f'val_{i}.png'))
                    plt.close()
            sdf_model.train()
            att_model.train()

        scheduler.step()
    print("Training completed!")

if __name__ == "__main__":
    args = parse_args()
    train(args)

