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


def render_image(rays, sdf_model, att_model, s, n_samples, chunk_size=4096, tau=None, att_model2=None):
    """Render a full image from rays.
    
    Args:
        rays: Ray tensor [H, W, 8] containing origins, directions, near, far
        sdf_model: SDF network (1M or 2M)
        att_model: First attenuation network (or only network for 1M-NeAS)
        s: Boundary sharpness parameter
        n_samples: Number of samples per ray
        chunk_size: Batch size for processing rays
        tau: Optional coarse-to-fine frequency parameter
        att_model2: Optional second attenuation network for 2M-NeAS
        
    Returns:
        Rendered intensity image [H, W]
    """
    from ..network import selector_function
    
    device = rays.device
    H, W, _ = rays.shape
    rays_flat = rays.reshape(-1, 8)
    
    is_2m = att_model2 is not None
    
    pred_intensities = []
    
    with torch.no_grad():
        for i in range(0, rays_flat.shape[0], chunk_size):
            chunk_rays = rays_flat[i:i+chunk_size]
            
            ray_origins = chunk_rays[..., :3]
            ray_directions = chunk_rays[..., 3:6] 
            near = chunk_rays[..., 6:7]
            far = chunk_rays[..., 7:8]
            n_chunk_rays = chunk_rays.shape[0]
            
            # Stratified sampling (no perturbation at eval time)
            t_vals = torch.linspace(0., 1., steps=n_samples, device=device)
            z_vals = near * (1. - t_vals) + far * t_vals  # [n_chunk_rays, n_samples]
            
            sampled_points = ray_origins.unsqueeze(1) + ray_directions.unsqueeze(1) * z_vals.unsqueeze(-1)
            sampled_points_flat = sampled_points.reshape(-1, 3)
            
            if is_2m:
                # 2M-NeAS: dual SDFs and dual attenuation
                d1, d2, feature_vector = sdf_model(sampled_points_flat, tau=tau)
                
                # Compute boundary values for both materials
                boundary_values1 = surface_boundary_function(d1, s)
                boundary_values2 = surface_boundary_function(d2, s)
                
                # Compute attenuation from both networks
                attenuation_values1 = att_model(feature_vector)
                attenuation_values2 = att_model2(feature_vector)
                
                # Apply boundary functions
                mu1 = attenuation_values1.squeeze(-1) * boundary_values1
                mu2 = attenuation_values2.squeeze(-1) * boundary_values2
                
                # Use selector function based on d2
                att_coeff = selector_function(d2, mu1, mu2)
            else:
                # 1M-NeAS: single SDF and single attenuation
                sdf_distances, feature_vector = sdf_model(sampled_points_flat, tau=tau)
                boundary_values = surface_boundary_function(sdf_distances, s)
                attenuation_values = att_model(feature_vector)
                
                att_coeff = attenuation_values.squeeze(-1) * boundary_values
            
            att_coeff = att_coeff.reshape(n_chunk_rays, n_samples)
            
            # Compute actual distances between adjacent samples (matching NAF)
            dists = z_vals[..., 1:] - z_vals[..., :-1]
            dists = torch.cat([dists, torch.ones_like(dists[..., :1]) * 1e-10], -1)
            dists = dists * torch.norm(ray_directions, dim=-1, keepdim=True)
            
            chunk_intensity = volume_render_intensity(att_coeff, dists)
            pred_intensities.append(chunk_intensity)
            
    pred_intensity = torch.cat(pred_intensities, dim=0)
    return pred_intensity.reshape(H, W)
