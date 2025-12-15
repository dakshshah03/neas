from mlp import FreqEncoder, att_freq_mlp, sdf_freq_mlp
from dataset import TIGREDataset
from torch.utils.data import DataLoader
import torch
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train NeAS model")
    
    parser.add_argument("--epochs", type=int, default=60, help="Number of epochs")
    parser.add_argument("--n_samples", type=int, default=64, help="Number of samples per ray")
    parser.add_argument("--lambda_reg", type=float, default=0.01, help="Regularization strength")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    
    parser.add_argument("--data_path", type=str, default="data/foot_50.pickle", help="Path to data file")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/", help="Directory to save checkpoints")
    parser.add_argument("--save_interval", type=int, default=10, help="Epoch interval for saving checkpoints")
    
    parser.add_argument("--feature_dim", type=int, default=8, help="Feature dimension for SDF model")
    
    return parser.parse_args()

def surface_boundary_function(d, s):
    return torch.exp(-s*d)/(1 + torch.exp(-s*d))

def volume_render_intensity(att_coeff, dists):
    mu_delta_sum = torch.sum(att_coeff * dists, dim=-1) # [batch_size, n_rays]
    I_hat = torch.exp(-mu_delta_sum) # [batch_size, n_rays]
    return I_hat

def train(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    data = TIGREDataset(path=args.data_path)
    train_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    sdf_model = sdf_freq_mlp(input_dim=3, output_dim=1, feature_dim=args.feature_dim).to(device)
    att_model = att_freq_mlp(input_dim=args.feature_dim, output_dim=1).to(device)

    # I do not know why the llm decided to make s a learnable parameter but lets see how that goes
    s = torch.tensor(20.0, requires_grad=True, device=device)

    optimizer = torch.optim.Adam(
        list(sdf_model.parameters()) + list(att_model.parameters()), 
        lr=args.lr
    )

    print("Starting training...")
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_int_loss = 0.0
        epoch_reg_loss = 0.0
        
        for batch_idx, batch_data in enumerate(train_loader):
            rays = batch_data['rays'].to(device)
            projs = batch_data['projs'].to(device)
            
            optimizer.zero_grad()
            
            # Extract ray components
            ray_origins = rays[..., :3]
            ray_directions = rays[..., 3:6] 
            near = rays[..., 6:7]
            far = rays[..., 7:8]
            
            batch_size, n_rays, _ = ray_origins.shape
            
            # ray point sampling stuff
            t_vals = torch.linspace(0., 1., steps=args.n_samples, device=device)
            sample_depths = near * (1. - t_vals) + far * t_vals
            sampled_points = ray_origins.unsqueeze(2) + ray_directions.unsqueeze(2) * sample_depths.unsqueeze(-1)
            sampled_points_flat = sampled_points.reshape(-1, 3)
            
            # Forward pass
            sdf_distances, feature_vector = sdf_model(sampled_points_flat)
            boundary_values = surface_boundary_function(sdf_distances, s)
            attenuation_values = torch.nn.functional.softplus(att_model(feature_vector))
            
            # attenuation coefficient
            att_coeff = attenuation_values.squeeze(-1) * boundary_values
            att_coeff = att_coeff.reshape(batch_size, n_rays, args.n_samples)
            
            # Volume rendering (equation 5)
            dists = (far - near) / args.n_samples
            pred_intensity = volume_render_intensity(att_coeff, dists)
            gt_intensity = torch.exp(-projs)
            
            # eikonal reg
            pts_eikonal = sampled_points_flat.clone().detach().requires_grad_(True)
            sdf_eikonal, _ = sdf_model(pts_eikonal)
            sdf_sum = sdf_eikonal.sum()
            
            # calculates the gradient of the SDF field 
            n = torch.autograd.grad(
                outputs=sdf_sum,
                inputs=pts_eikonal,
                create_graph=True,
            )[0]
            
            # loss (int and reg)
            L_int = torch.nn.functional.mse_loss(pred_intensity, gt_intensity)
            n_norm = torch.linalg.norm(n, dim=-1)
            L_reg = torch.mean((n_norm - 1.0)**2)
            L_total = L_int + args.lambda_reg * L_reg
            
            L_total.backward()
            
            optimizer.step()
            
            epoch_int_loss += L_int.item()
            epoch_reg_loss += L_reg.item()
            epoch_loss += L_total.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}: "
                      f"Total Loss: {L_total.item():.6f}, "
                      f"Int Loss: {L_int.item():.6f}, "
                      f"Reg Loss: {L_reg.item():.6f}")
        
        # Print epoch summary
        avg_loss = epoch_loss / len(train_loader)
        avg_int_loss = epoch_int_loss / len(train_loader)
        avg_reg_loss = epoch_reg_loss / len(train_loader)
        
        print(f"Epoch {epoch+1}/{args.epochs} Summary - "
              f"Avg Loss: {avg_loss:.6f}, "
              f"Avg Int Loss: {avg_int_loss:.6f}, "
              f"Avg Reg Loss: {avg_reg_loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'sdf_model_state_dict': sdf_model.state_dict(),
                'att_model_state_dict': att_model.state_dict(),
                's': s,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            print(f"Checkpoint saved at epoch {epoch+1}")

    print("Training completed!")

if __name__ == "__main__":
    args = parse_args()
    train(args)

