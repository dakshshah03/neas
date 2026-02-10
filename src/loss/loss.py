import torch


def calc_mse_loss(loss, x, y):
    """
    Calculate mse loss.
    """
    loss_mse = torch.mean((x-y)**2)
    loss["loss"] += loss_mse
    loss["loss_mse"] = loss_mse
    return loss


def calc_eikonal_loss(loss, sdf_distances, sampled_points, lambda_reg):
    """
    Calculate eikonal regularization loss.
    
    Args:
        loss: Dictionary to store losses
        sdf_distances: SDF values at sampled points
        sampled_points: Points where SDF was evaluated
        lambda_reg: Regularization weight
    """
    pts_eikonal = sampled_points.clone().detach().requires_grad_(True)
    sdf_sum = sdf_distances.sum()
    
    # Calculate gradient of SDF field
    n = torch.autograd.grad(
        outputs=sdf_sum,
        inputs=pts_eikonal,
        create_graph=True,
    )[0]
    
    # Eikonal loss: ||grad(SDF)|| should be 1
    n_norm = torch.linalg.norm(n, dim=-1)
    L_reg = torch.mean((n_norm - 1.0)**2)
    
    loss["loss"] += lambda_reg * L_reg
    loss["loss_reg"] = lambda_reg * L_reg
    return loss
