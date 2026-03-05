import torch


def surface_boundary_function(d, s):
    """Compute surface boundary values from SDF distances.

    Implements Ω(d, s) = exp(-s·d) / (1 + exp(-s·d)) from the paper (Eq. 2),
    which is algebraically identical to sigmoid(-s·d).  Using torch.sigmoid
    is preferred because PyTorch's numerically-stable log-sum-exp path avoids
    overflow when s·d is large and positive.

    Args:
        d: SDF distance values
        s: boundary sharpness parameter

    Returns:
        Boundary values in (0, 1); ≈1 inside surface (d<0), ≈0 outside (d>0)
    """
    return torch.sigmoid(-s * d)


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


def render_image(rays, sdf_model, att_model, s, n_samples, chunk_size=4096, tau=None, num_materials=1):
    """Render a full image from rays.
    
    Args:
        rays: Ray tensor [H, W, 8] containing origins, directions, near, far
        sdf_model: SDF network (1M or KM)
        att_model: Attenuation network (nn.Sequential for 1M, SharedAttenuationMLP for KM)
        s: Boundary sharpness parameter
        n_samples: Number of samples per ray
        chunk_size: Batch size for processing rays
        tau: Optional coarse-to-fine frequency parameter
        num_materials: 1 for 1M-NeAS, >=2 for KM-NeAS
        
    Returns:
        Rendered intensity image [H, W]
    """
    from ..network import nested_material_selector
    
    device = rays.device
    H, W, _ = rays.shape
    rays_flat = rays.reshape(-1, 8)
    
    is_km = num_materials > 1
    
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
            
            if is_km:
                # KM-NeAS: K SDFs + shared attenuation + nested selector
                distances, feature_vector = sdf_model(sampled_points_flat, tau=tau)
                boundary_values = [surface_boundary_function(d, s) for d in distances]
                raw_attenuations = att_model(feature_vector)
                att_coeff = nested_material_selector(boundary_values, raw_attenuations)
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
