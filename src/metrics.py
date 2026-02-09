import os
import torch
import numpy as np
from PIL import Image
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import pandas as pd
import argparse

from dataset import TIGREDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import re

from train import render_image
from mlp import create_neas_model

from mesh import extract_mesh_from_sdf, _load_sdf_model_from_checkpoint

# 1 load validation dataset
# 2 run inference on validation dataset
# 3 save images for GT and validation in separate directories, return those directories

def get_predictions(model_path, data_dir, save_dir, device):
    """
    loads pickle file of TIGRE formatted data

    Args:
        model_path (str): path to trained SDF/attenuation model
        data_dir (str): path to pickle file containing validation data
        save_dir (str): path to directory to save images
        device (str, optional): device to run inference on. Defaults to 'cuda' if available else 'cpu'.

    Returns:
        pred_images (list[PIL.Image]): list of PIL Images of model predictions 
        gt_images (list[PIL.Image]): list of PIL Images of the ground truth
    """
    checkpoint = torch.load(model_path, map_location=device)
        
    encoding = checkpoint.get('encoding', 'frequency')
    feature_dim = checkpoint.get('feature_dim', 8)
    alpha = checkpoint.get('alpha', 1.0)
    beta = checkpoint.get('beta', 0.0)
    multires = checkpoint.get('multires', 6)
    num_levels = checkpoint.get('num_levels', 14)
    level_dim = checkpoint.get('level_dim', 2)
    base_resolution = checkpoint.get('base_resolution', 16)
    log2_hashmap_size = checkpoint.get('log2_hashmap_size', 19)
    
    # Create models
    sdf_model, att_model = create_neas_model(
        encoding=encoding,
        feature_dim=feature_dim,
        alpha=alpha,
        beta=beta,
        multires=multires,
        num_levels=num_levels,
        level_dim=level_dim,
        base_resolution=base_resolution,
        log2_hashmap_size=log2_hashmap_size
    )
    sdf_model = sdf_model.to(device)
    att_model = att_model.to(device)
    
    sdf_model.load_state_dict(checkpoint['sdf_model_state_dict'])
    att_model.load_state_dict(checkpoint['att_model_state_dict'])
    
    s = checkpoint['s']
    
    sdf_model.eval()
    att_model.eval()

    os.makedirs(save_dir, exist_ok=True)
    gt_dir = os.path.join(save_dir, 'gt')
    pred_dir = os.path.join(save_dir, 'pred')

    # loads images if dir alr exists
    if os.path.exists(pred_dir) and os.path.exists(gt_dir) and len(os.listdir(pred_dir)) > 0 and len(os.listdir(gt_dir)) > 0:
        pred_images = [Image.open(os.path.join(pred_dir, fname)) for fname in sorted(os.listdir(pred_dir)) if fname.endswith('.png')]
        gt_images = [Image.open(os.path.join(gt_dir, fname)) for fname in sorted(os.listdir(gt_dir)) if fname.endswith('.png')]
        return pred_images, gt_images

    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    val_data = TIGREDataset(path=data_dir, type="val")
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)

    pred_images = []
    gt_images = []
    for i, batch in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
        rays = batch['rays'].squeeze(0).to(device)  # [H, W, 8]
        projs = batch['projs'].squeeze(0).to(device)  # [H, W]

        with torch.no_grad():
            pred_img = render_image(rays, sdf_model, att_model, s, n_samples=128, chunk_size=4096)

        pred_img_np = (pred_img.cpu().numpy().T * 255).clip(0, 255).astype(np.uint8)
        pred_pil = Image.fromarray(pred_img_np)
        pred_pil.save(os.path.join(pred_dir, f"pred_{i}.png"))
        pred_images.append(pred_pil)

        gt_img_np = (torch.exp(-projs).cpu().numpy().T * 255).clip(0, 255).astype(np.uint8)
        gt_pil = Image.fromarray(gt_img_np)
        gt_pil.save(os.path.join(gt_dir, f"gt_{i}.png"))
        gt_images.append(gt_pil)
    return pred_images, gt_images

def compute_lpips(pred_images, gt_images, device):
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
    lpips_scores = []
    
    for pred, gt in zip(pred_images, gt_images):
        pred_tensor = torch.from_numpy(np.array(pred)).float().div(255).unsqueeze(0)
        pred_tensor = pred_tensor.repeat(3,1,1)
        pred_tensor = pred_tensor.unsqueeze(0).to(device)
        
        gt_tensor = torch.from_numpy(np.array(gt)).float().div(255).unsqueeze(0)
        gt_tensor = gt_tensor.repeat(3,1,1)
        gt_tensor = gt_tensor.unsqueeze(0).to(device)
        
        score = lpips(pred_tensor, gt_tensor).item()
        
        lpips_scores.append(score)
    avg_lpips = float(np.mean(lpips_scores)) if lpips_scores else float('nan')
    return lpips_scores, avg_lpips

def compute_ssim(pred_images, gt_images, device):
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    ssim_scores = []
    for pred, gt in zip(pred_images, gt_images):
        pred_np = np.array(pred)
        gt_np = np.array(gt)
        # If grayscale, add channel dim
        if pred_np.ndim == 2:
            pred_np = pred_np[..., None]
        if gt_np.ndim == 2:
            gt_np = gt_np[..., None]
                
        pred_tensor = torch.from_numpy(pred_np).float().div(255).permute(2,0,1).unsqueeze(0)
        gt_tensor = torch.from_numpy(gt_np).float().div(255).permute(2,0,1).unsqueeze(0)
        
        if pred_tensor.shape[1] == 1:
            pred_tensor = pred_tensor.repeat(1,3,1,1)
        if gt_tensor.shape[1] == 1:
            gt_tensor = gt_tensor.repeat(1,3,1,1)
            
        pred_tensor = pred_tensor.to(device)
        gt_tensor = gt_tensor.to(device)
        score = ssim(pred_tensor, gt_tensor).item()
        ssim_scores.append(score)
    avg_ssim = float(np.mean(ssim_scores)) if ssim_scores else float('nan')
    return ssim_scores, avg_ssim
    
def compute_psnr(pred_images, gt_images, device):
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    psnr_scores = []
    for pred, gt in zip(pred_images, gt_images):
        pred_np = np.array(pred)
        gt_np = np.array(gt)
        
        if pred_np.ndim == 2:
            pred_np = pred_np[..., None]
        if gt_np.ndim == 2:
            gt_np = gt_np[..., None]
            
        pred_tensor = torch.from_numpy(pred_np).float().div(255).permute(2,0,1).unsqueeze(0)
        gt_tensor = torch.from_numpy(gt_np).float().div(255).permute(2,0,1).unsqueeze(0)
        
        if pred_tensor.shape[1] == 1:
            pred_tensor = pred_tensor.repeat(1,3,1,1)
        if gt_tensor.shape[1] == 1:
            gt_tensor = gt_tensor.repeat(1,3,1,1)
        pred_tensor = pred_tensor.to(device)
        gt_tensor = gt_tensor.to(device)
        score = psnr(pred_tensor, gt_tensor).item()
        psnr_scores.append(score)
    avg_psnr = float(np.mean(psnr_scores)) if psnr_scores else float('nan')
    return psnr_scores, avg_psnr

def save_metrics(save_dir, lpips_scores, ssim_scores, psnr_scores):
    metrics = {
        'LPIPS': lpips_scores,
        'SSIM': ssim_scores,
        'PSNR': psnr_scores
    }
    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(save_dir, 'validation_metrics.csv'), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NeAS model validation and save predictions.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--val_pickle', type=str, required=True, help='Path to validation pickle file')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save predictions (default: checkpoints folder of model)')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu, default: auto)')
    args = parser.parse_args()

    if args.save_dir is None:
        model_dir = os.path.dirname(os.path.abspath(args.model_path))
        
        checkpoint_base = os.path.basename(args.model_path)
        match = re.search(r'epoch_(\d+)', checkpoint_base)
        if match:
            epoch_str = match.group(1)
            val_folder = f'validation_and_mesh_epoch_{epoch_str}'
        else:
            val_folder = 'validation_and_mesh_results'
        args.save_dir = os.path.join(model_dir, val_folder)

    if args.device is not None:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Model: {args.model_path}\nValidation data: {args.val_pickle}\nSave dir: {args.save_dir}\nDevice: {device}")
    preds, gt = get_predictions(args.model_path, args.val_pickle, args.save_dir, device=device)
    
    # TODO: compute and save metrics
    lpips_scores, avg_lpips = compute_lpips(preds, gt, device=device)
    print(f"Average LPIPS: {avg_lpips:.4f}")
    
    ssim_scores, avg_ssim = compute_ssim(preds, gt, device=device)
    print(f"Average SSIM: {avg_ssim:.4f}")
    
    psnr_scores, avg_psnr = compute_psnr(preds, gt, device=device)
    print(f"Average PSNR: {avg_psnr:.4f}")
    
    save_metrics(args.save_dir, lpips_scores, ssim_scores, psnr_scores)

    checkpoint_path = args.model_path
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    feature_dim = checkpoint.get('feature_dim', 8)
    sdf_model, _ = _load_sdf_model_from_checkpoint(checkpoint_path, feature_dim=feature_dim, device=device)

    bounds = ((-1, -1, -1), (1, 1, 1))
    resolution = 256
    level = 0.0
    batch_size = 65536

    mesh_path = os.path.join(args.save_dir, "mesh.ply")
    print(f"Extracting mesh to: {mesh_path}")
    extract_mesh_from_sdf(
        sdf_model,
        bounds=bounds,
        resolution=resolution,
        level=level,
        device=device,
        batch_size=batch_size,
        save_path=mesh_path
    )
    print(f"Mesh saved to: {mesh_path}")

