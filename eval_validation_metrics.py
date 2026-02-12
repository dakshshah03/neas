#!/usr/bin/env python3
"""
Simple evaluation script (keeps separate from existing `src/metrics.py`).

Usage:
    python eval_validation_metrics.py --checkpoint checkpoints/xxx/checkpoint_epoch_100.pth \
        --val_pickle data/raw/your_val.pickle --device cuda

Outputs (printed + optional CSV):
 - 3D PSNR (average over volume)
 - 3D SSIM (average over volume)
 - 2D projection PSNR (mean over validation views)
 - 2D projection SSIM (mean over validation views)

The script re-uses the project's models/renderers and the TIGRE dataset loader.
"""
import os
import sys
import argparse
import csv
import torch
import numpy as np

# Ensure `src` modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dataset.tigre import TIGREDataset
from render.render import render_image, surface_boundary_function
from network import (
    sdf_freq_mlp, att_freq_mlp,
    sdf_hash_mlp, att_hash_mlp, selector_function
)
from utils.util import get_psnr, get_ssim, get_psnr_3d, get_ssim_3d


def _build_models_from_checkpoint(ckpt, device):
    feature_dim = ckpt.get('feature_dim', 8)
    encoding_type = ckpt.get('encoding_type', 'freq')
    num_materials = ckpt.get('num_materials', 1)

    alpha1 = ckpt.get('alpha1', ckpt.get('alpha', 3.4))
    beta1 = ckpt.get('beta1', ckpt.get('beta', 0.1))

    s_param = ckpt.get('s', 20.0)

    if encoding_type == 'freq':
        if num_materials == 1:
            sdf_model = sdf_freq_mlp(input_dim=3, output_dim=1, feature_dim=feature_dim).to(device)
        else:
            sdf_model = sdf_freq_mlp(input_dim=3, output_dim=2, feature_dim=feature_dim).to(device)
        att_model1 = att_freq_mlp(input_dim=feature_dim, output_dim=1, alpha=alpha1, beta=beta1).to(device)
        att_model2 = None
    else:  # hash
        num_levels = ckpt.get('num_levels', 14)
        level_dim = ckpt.get('level_dim', 2)
        base_resolution = ckpt.get('base_resolution', 16)
        log2_hashmap_size = ckpt.get('log2_hashmap_size', 19)

        if num_materials == 1:
            sdf_model = sdf_hash_mlp(input_dim=3, output_dim=1, feature_dim=feature_dim,
                                     num_levels=num_levels, level_dim=level_dim,
                                     base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size).to(device)
        else:
            sdf_model = sdf_hash_mlp(input_dim=3, output_dim=2, feature_dim=feature_dim,
                                     num_levels=num_levels, level_dim=level_dim,
                                     base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size).to(device)

        att_model1 = att_hash_mlp(input_dim=feature_dim, output_dim=1,
                                  num_levels=num_levels, level_dim=level_dim,
                                  base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size,
                                  alpha=alpha1, beta=beta1).to(device)
        att_model2 = None

    # Load weights (be permissive with key names used across checkpoints)
    sdf_state = None
    if 'sdf_model_state_dict' in ckpt:
        sdf_state = ckpt['sdf_model_state_dict']
    elif 'sdf_state_dict' in ckpt:
        sdf_state = ckpt['sdf_state_dict']

    if sdf_state is not None:
        sdf_model.load_state_dict(sdf_state)

    # attenuation weights: support 'att_model1_state_dict' or older 'att_model_state_dict'
    att_state = None
    if 'att_model1_state_dict' in ckpt:
        att_state = ckpt['att_model1_state_dict']
    elif 'att_model_state_dict' in ckpt:
        att_state = ckpt['att_model_state_dict']

    if att_state is not None:
        att_model1.load_state_dict(att_state)

    # second attenuation net for 2M
    if num_materials == 2:
        if 'att_model2_state_dict' in ckpt:
            att_model2 = att_hash_mlp(input_dim=feature_dim, output_dim=1,
                                      num_levels=num_levels, level_dim=level_dim,
                                      base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size,
                                      alpha=ckpt.get('alpha2', 5.5), beta=ckpt.get('beta2', 3.5)).to(device)
            att_model2.load_state_dict(ckpt['att_model2_state_dict'])

    return sdf_model, att_model1, att_model2, float(s_param), int(num_materials)


def sample_3d_volume_from_models(sdf_model, att_model1, att_model2, s_param, voxels, num_materials, chunk_size=8192, device='cuda'):
    # voxels: torch tensor [n1, n2, n3, 3]
    n1, n2, n3, _ = voxels.shape
    total = n1 * n2 * n3
    voxels_flat = voxels.reshape(-1, 3).to(device)

    pred_atten = []
    with torch.no_grad():
        for i in range(0, total, chunk_size):
            chunk = voxels_flat[i:i+chunk_size]
            if num_materials == 1:
                sdf_d, feature = sdf_model(chunk, tau=None)
                boundary = surface_boundary_function(sdf_d, torch.tensor(s_param, device=device))
                att_vals = att_model1(feature)
                att_coeff = att_vals.squeeze(-1) * boundary
            else:
                d1, d2, feature = sdf_model(chunk, tau=None)
                b1 = surface_boundary_function(d1, torch.tensor(s_param, device=device))
                b2 = surface_boundary_function(d2, torch.tensor(s_param, device=device))
                mu1 = att_model1(feature).squeeze(-1) * b1
                mu2 = att_model2(feature).squeeze(-1) * b2
                att_coeff = selector_function(d2, mu1, mu2)

            pred_atten.append(att_coeff.cpu())

    pred_atten = torch.cat(pred_atten, dim=0)
    pred_volume = pred_atten.reshape(n1, n2, n3).numpy()
    return pred_volume


def main():
    parser = argparse.ArgumentParser(description="Compute average 2D/3D metrics over validation views")
    parser.add_argument('--checkpoint', '-c', required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--val_pickle', '-p', required=True, help='Path to TIGRE-format pickle (validation inside)')
    parser.add_argument('--device', '-d', default=None, help='cuda or cpu (default auto)')
    parser.add_argument('--n_samples', type=int, default=128, help='Number of samples for rendering (2D)')
    parser.add_argument('--chunk_size', type=int, default=4096, help='Chunk size for rendering / 3D sampling')
    parser.add_argument('--save_csv', type=str, default=None, help='Optional CSV file to write per-view metrics')
    args = parser.parse_args()

    device = args.device if args.device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Checkpoint: {args.checkpoint}\nVal pickle: {args.val_pickle}\nDevice: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    sdf_model, att_model1, att_model2, s_param, num_materials = _build_models_from_checkpoint(ckpt, device)
    sdf_model.eval(); att_model1.eval();
    if att_model2 is not None:
        att_model2.eval()

    # Load validation data
    val_ds = TIGREDataset(args.val_pickle, n_rays=1024, type='val', device=device)

    n_views = val_ds.n_samples
    print(f"Validation views found: {n_views}")

    proj_psnrs = []
    proj_ssims = []

    # loop over validation views
    for i in range(n_views):
        rays = val_ds.rays[i].to(device)       # [H, W, 8]
        projs = val_ds.projs[i].to(device)     # [H, W]

        with torch.no_grad():
            pred_img = render_image(rays, sdf_model, att_model1, torch.tensor(s_param, device=device), args.n_samples, chunk_size=args.chunk_size, tau=None, att_model2=att_model2)

        # GT projection is exp(-projs)
        gt_proj = torch.exp(-projs)

        p_psnr = get_psnr(pred_img, gt_proj)
        p_ssim = get_ssim(pred_img, gt_proj)

        proj_psnrs.append(float(p_psnr.item()))
        proj_ssims.append(float(p_ssim))

        print(f"View {i+1:02d}/{n_views:02d}  --  Proj PSNR: {proj_psnrs[-1]:.4f}, Proj SSIM: {proj_ssims[-1]:.4f}")

    # 3D sampling
    print("Sampling full 3D volume from model (this may take a while)...")
    pred_volume = sample_3d_volume_from_models(sdf_model, att_model1, att_model2, s_param, val_ds.voxels, num_materials, chunk_size=8192, device=device)
    gt_volume = val_ds.image.cpu().numpy()

    vol_psnr = float(get_psnr_3d(pred_volume, gt_volume))
    vol_ssim = float(get_ssim_3d(pred_volume, gt_volume))

    print('\n=== Summary ===')
    print(f"Average 2D projection PSNR : {np.mean(proj_psnrs):.4f}")
    print(f"Average 2D projection SSIM : {np.mean(proj_ssims):.4f}")
    print(f"3D volume PSNR            : {vol_psnr:.4f}")
    print(f"3D volume SSIM            : {vol_ssim:.4f}")

    if args.save_csv:
        rows = []
        for idx, (p_psnr, p_ssim) in enumerate(zip(proj_psnrs, proj_ssims)):
            rows.append({'view': idx, 'proj_psnr': p_psnr, 'proj_ssim': p_ssim})
        rows.append({'view': 'avg', 'proj_psnr': float(np.mean(proj_psnrs)), 'proj_ssim': float(np.mean(proj_ssims))})
        rows.append({'view': 'vol', 'proj_psnr': vol_psnr, 'proj_ssim': vol_ssim})
        keys = rows[0].keys()
        with open(args.save_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Per-view metrics + summary written to: {args.save_csv}")


if __name__ == '__main__':
    main()
