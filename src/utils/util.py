import numpy as np
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio


def get_psnr(pred, gt):
    """Compute PSNR between predicted and ground truth images."""
    device = pred.device
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    # Convert to [1, C, H, W] format
    if pred.dim() == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
        gt = gt.unsqueeze(0).unsqueeze(0)
    return psnr_metric(pred, gt).item()


def get_mse(pred, gt):
    """Compute MSE between predicted and ground truth."""
    return torch.mean((pred - gt) ** 2).item()


def get_ssim_3d(pred, gt):
    """Compute SSIM for 3D volumes (computes per-slice and averages)."""
    device = pred.device
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    # Compute SSIM for each slice along the last dimension
    n_slices = pred.shape[-1]
    ssim_scores = []
    
    for i in range(n_slices):
        pred_slice = pred[..., i].unsqueeze(0).unsqueeze(0)
        gt_slice = gt[..., i].unsqueeze(0).unsqueeze(0)
        ssim_scores.append(ssim_metric(pred_slice, gt_slice).item())
    
    return np.mean(ssim_scores)


def get_psnr_3d(pred, gt):
    """Compute PSNR for 3D volumes."""
    device = pred.device
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    return psnr_metric(pred.unsqueeze(0).unsqueeze(0), gt.unsqueeze(0).unsqueeze(0)).item()


def cast_to_image(tensor):
    """Convert tensor to numpy image in [0, 1] range."""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    
    # Normalize to [0, 1]
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
    
    # Add channel dimension if grayscale
    if tensor.ndim == 2:
        tensor = tensor[..., np.newaxis]
    
    return tensor
