import torch


def surface_boundary_function(d, s):
    """Compute surface boundary values from SDF distances.
    
    Args:
        d: SDF distance values
        s: boundary sharpness parameter
        
    Returns:
        Boundary values exp(-s*d)/(1 + exp(-s*d))
    """
    return torch.exp(-s*d)/(1 + torch.exp(-s*d))


def volume_render_intensity(att_coeff, dists):
    """Volume render intensity using attenuation coefficients.
    
    Args:
        att_coeff: Attenuation coefficients [batch_size, n_rays, n_samples] or [batch_size, n_samples]
        dists: Distance deltas between samples
        
    Returns:
        I_hat: Rendered intensity exp(-sum(mu*delta))
    """
    mu_delta_sum = torch.sum(att_coeff * dists, dim=-1)
    I_hat = torch.exp(-mu_delta_sum)
    return I_hat


def render_image(rays, sdf_model, att_model, s, n_samples, chunk_size=4096, tau=None):
    """Render a full image from rays.
    
    Args:
        rays: Ray tensor [H, W, 8] containing origins, directions, near, far
        sdf_model: SDF network
        att_model: Attenuation network
        s: Boundary sharpness parameter
        n_samples: Number of samples per ray
        chunk_size: Batch size for processing rays
        tau: Optional coarse-to-fine frequency parameter
        
    Returns:
        Rendered intensity image [H, W]
    """
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
            # Attenuation model now includes custom activation (alpha*sigmoid(x) + beta)
            attenuation_values = att_model(feature_vector)
            
            att_coeff = attenuation_values.squeeze(-1) * boundary_values
            att_coeff = att_coeff.reshape(chunk_rays.shape[0], n_samples)
            
            dists = (far - near) / n_samples
            chunk_intensity = volume_render_intensity(att_coeff, dists)
            pred_intensities.append(chunk_intensity)
            
    pred_intensity = torch.cat(pred_intensities, dim=0)
    return pred_intensity.reshape(H, W)
