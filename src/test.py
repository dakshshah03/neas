from mlp import FreqEncoder, att_freq_mlp, sdf_freq_mlp

from dataset import TIGREDataset
from torch.utils.data import DataLoader
import torch

def surface_boundary_function(d, s=10):
    return torch.exp(-s*d)/(1 + torch.exp(-s*d))

# load data, test 
data = TIGREDataset(
    path="data/foot_50.pickle"
)

train_loader = DataLoader(
    data, 
    batch_size=1,     
    shuffle=True,
    num_workers=0      
)

# sdf mlp
sdf_model = sdf_freq_mlp(input_dim=3, output_dim=1, feature_dim=8)
att_model = att_freq_mlp(input_dim=8, output_dim=1)  # Adjust dims as needed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sdf_model = sdf_model.to(device)
att_model = att_model.to(device)
print(f"Using device: {device}")

for batch_idx, data in enumerate(train_loader):
    rays = data['rays']     # shape: [batch_size, n_rays, 8] (origin + direction + near + far)
    projs = data['projs']   # shape: [batch_size, n_rays] (projection values)
    
    # Move data to device
    rays = rays.to(device)
    projs = projs.to(device)
    
    print(f"Batch {batch_idx}:")
    print(f"Rays shape: {rays.shape}")
    print(f"Projections shape: {projs.shape}")
    
    # mildly vibe coded ray extraction and sampling
    
    # Extract ray origins and directions for SDF MLP testing
    # Assuming rays format: [origin_x, origin_y, origin_z, dir_x, dir_y, dir_z, near, far]
    ray_origins = rays[..., :3]  # [batch_size, n_rays, 3]
    ray_directions = rays[..., 3:6]  # [batch_size, n_rays, 3]
    
    # Sample points along rays for SDF evaluation
    # For testing, let's just use ray origins as 3D positions
    batch_size, n_rays, _ = ray_origins.shape
    test_positions = ray_origins.reshape(-1, 3)  # [batch_size * n_rays, 3]
    
    print(f"Test positions shape: {test_positions.shape}")
    
    # Forward pass through sdf mlp for testing
    with torch.no_grad():
        sdf_distances, feature_vector = sdf_model(test_positions)
        boundary_values = surface_boundary_function(sdf_distances)
        attenuation_values = att_model(feature_vector)
        att_coeff = attenuation_values * boundary_values
     
    print(f"Feature vector shape: {feature_vector.shape}")   # [1024, 8] aka [b, k]
    print(f"SDF distances shape: {sdf_distances.shape}")  # [1024] aka [b]
    print(f"SDF distances range: [{sdf_distances.min().item():.4f}, {sdf_distances.max().item():.4f}]")
    print(f"Boundary values shape: {boundary_values.shape}")
    print(f"Attenuation values shape: {attenuation_values.shape}")
    print(f"Attenuation coefficient shape: {att_coeff.shape}")
    
    break

