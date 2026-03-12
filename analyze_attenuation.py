#!/usr/bin/env python3
"""
Analyze attenuation distribution from trained 1M-NeAS model.

This script helps determine optimal alpha/beta parameters for 2M-NeAS
by analyzing the attenuation distribution from a 1M-NeAS checkpoint.

Usage:
    python analyze_attenuation.py --checkpoint checkpoints/foot_50_1m_hash/checkpoint_epoch_500.pth \
                                   --config config/foot_configs/foot_50_1m_hash.yaml \
                                   --output attenuation_analysis_foot.png
"""

import argparse
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pathlib import Path

from src.dataset import TIGREDataset
from src.network import sdf_freq_mlp, sdf_hash_mlp, att_freq_mlp, att_hash_mlp
from src.render import surface_boundary_function


def load_model(checkpoint_path, config_path, device='cuda'):
    """Load trained 1M-NeAS model from checkpoint."""
    
    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Verify it's 1M
    num_materials = cfg["network"].get("num_materials", 1)
    if num_materials != 1:
        raise ValueError(f"This script requires a 1M-NeAS checkpoint, but config has num_materials={num_materials}")
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Create networks
    feature_dim = cfg["network"].get("feature_dim", 8)
    multires = cfg["network"].get("multires", 6)
    encoding_type = cfg["network"].get("encoding_type", "freq")
    alpha = cfg["network"].get("alpha", 3.4)
    beta = cfg["network"].get("beta", 0.1)
    
    if encoding_type == "freq":
        sdf_model = sdf_freq_mlp(input_dim=3, output_dim=1, feature_dim=feature_dim, multires=multires).to(device)
        att_model = att_freq_mlp(input_dim=feature_dim, output_dim=1, alpha=alpha, beta=beta).to(device)
    elif encoding_type == "hash":
        num_levels = cfg["network"].get("num_levels", 14)
        level_dim = cfg["network"].get("level_dim", 2)
        base_resolution = cfg["network"].get("base_resolution", 16)
        log2_hashmap_size = cfg["network"].get("log2_hashmap_size", 19)
        
        sdf_model = sdf_hash_mlp(input_dim=3, output_dim=1, feature_dim=feature_dim,
                                num_levels=num_levels, level_dim=level_dim,
                                base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size).to(device)
        att_model = att_hash_mlp(input_dim=feature_dim, output_dim=1,
                                num_levels=num_levels, level_dim=level_dim,
                                base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size,
                                alpha=alpha, beta=beta).to(device)
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")
    
    # Load weights
    sdf_model.load_state_dict(ckpt["sdf_model_state_dict"])
    if "att_model1_state_dict" in ckpt:
        att_model.load_state_dict(ckpt["att_model1_state_dict"])
    elif "att_model_state_dict" in ckpt:
        att_model.load_state_dict(ckpt["att_model_state_dict"])
    
    s = ckpt["s"]
    
    sdf_model.eval()
    att_model.eval()
    
    return sdf_model, att_model, s, cfg


def sample_volume(sdf_model, att_model, s, voxels, chunk_size=8192, device='cuda'):
    """Sample 3D volume of attenuation coefficients."""
    
    n1, n2, n3, _ = voxels.shape
    total_voxels = n1 * n2 * n3
    
    voxels_flat = voxels.reshape(-1, 3)
    voxels_flat = torch.tensor(voxels_flat, dtype=torch.float32, device=device)
    
    pred_attenuation = []
    
    print(f"Sampling {total_voxels:,} voxels...")
    with torch.no_grad():
        for i in range(0, total_voxels, chunk_size):
            chunk_voxels = voxels_flat[i:i+chunk_size]
            
            sdf_distances, feature_vector = sdf_model(chunk_voxels, tau=None)
            boundary_values = surface_boundary_function(sdf_distances, s)
            attenuation_values = att_model(feature_vector)
            
            att_coeff = attenuation_values.squeeze(-1) * boundary_values
            pred_attenuation.append(att_coeff.cpu())
            
            if (i // chunk_size) % 10 == 0:
                print(f"  Progress: {i}/{total_voxels} ({100*i/total_voxels:.1f}%)")
    
    pred_attenuation = torch.cat(pred_attenuation, dim=0)
    pred_volume = pred_attenuation.reshape(n1, n2, n3).numpy()
    
    print(f"Volume sampled: shape={pred_volume.shape}, "
          f"range=[{pred_volume.min():.6f}, {pred_volume.max():.6f}]")
    
    return pred_volume


def analyze_distribution(volume, output_path=None, title="Attenuation Distribution"):
    """Analyze attenuation distribution and suggest alpha/beta values.

    For single-material (1M) distributions we recommend using beta=0.1 and
    setting alpha so the activation covers from 0.1 up to the maximum
    attenuation value. For bimodal (2M) cases the script computes a
    threshold between the two peaks and generates relu-modifier parameters
    where
        output = alpha * relu(input) + beta
    with beta1 fixed at 0.1, beta2 = 0.1 + alpha1 (i.e. the threshold), and
    alpha2 set so that the material 2 range spans up to the maximum
    attenuation. These rules ensure the alpha/beta ranges cover the
    entire observed attenuation range for each material.
    """
    
    # Flatten and remove very small values (air/background)
    values = volume.flatten()
    threshold = 0.01  # Remove near-zero values
    values_nonzero = values[values > threshold]
    
    print(f"\n{'='*60}")
    print(f"ATTENUATION DISTRIBUTION ANALYSIS")
    print(f"{'='*60}")
    print(f"Total voxels: {len(values):,}")
    print(f"Non-zero voxels (> {threshold}): {len(values_nonzero):,}")
    print(f"Overall range: [{values.min():.4f}, {values.max():.4f}]")
    print(f"Non-zero range: [{values_nonzero.min():.4f}, {values_nonzero.max():.4f}]")
    print(f"Mean (non-zero): {values_nonzero.mean():.4f}")
    print(f"Median (non-zero): {np.median(values_nonzero):.4f}")
    print(f"Std (non-zero): {values_nonzero.std():.4f}")
    
    # Percentiles
    print(f"\nPercentiles (non-zero values):")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(values_nonzero, p)
        print(f"  {p:2d}th: {val:.4f}")
    
    # Create histogram and find peaks
    hist, bin_edges = np.histogram(values_nonzero, bins=100)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Find peaks in histogram (potential material modes)
    peaks, properties = find_peaks(hist, prominence=len(values_nonzero)*0.01, distance=5)
    
    print(f"\nDetected {len(peaks)} peak(s) in histogram:")
    for i, peak_idx in enumerate(peaks):
        peak_value = bin_centers[peak_idx]
        peak_count = hist[peak_idx]
        print(f"  Peak {i+1}: attenuation = {peak_value:.4f} (count: {peak_count:,})")
    
    # Suggestions for NeAS parameters (1M or 2M)
    print(f"\n{'='*60}")
    print(f"RECOMMENDATIONS FOR NeAS PARAMETERS (1M/2M)")
    print(f"{'='*60}")
    
    if len(peaks) >= 2:
        # Two or more peaks detected - likely muscle and bone
        peak1_val = bin_centers[peaks[0]]
        peak2_val = bin_centers[peaks[1]]
        
        # Find valley between peaks
        valley_start = peaks[0]
        valley_end = peaks[1]
        valley_idx = valley_start + np.argmin(hist[valley_start:valley_end])
        threshold_val = bin_centers[valley_idx]
        
        print(f"\nDetected bimodal distribution (muscle + bone):")
        print(f"  Material 1 peak (muscle): {peak1_val:.4f}")
        print(f"  Material 2 peak (bone):   {peak2_val:.4f}")
        print(f"  Suggested threshold:      {threshold_val:.4f}")
        
        # Calculate suggested parameters using relu-mod formula:
        # alpha * relu() + beta
        # beta1 fixed at 0.1, alpha1 covers from beta1 up to threshold
        # beta2 = 0.1 + alpha1 (which equals threshold)
        # alpha2 covers from beta2 to max attenuation
        max_val = values_nonzero.max()
        beta1 = 0.01
        alpha1 = max(threshold_val - beta1, 0.0)
        beta2 = beta1 + alpha1
        alpha2 = max_val - beta2
        if alpha2 < 0:
            alpha2 = 0.0
        
        print(f"\nSuggested config parameters (relu modifiers):")
        print(f"  # Material 1 (soft tissue/muscle): beta1={beta1:.2f}, alpha1={alpha1:.2f}")
        print(f"     -> covers range [{beta1:.2f}, {beta1+alpha1:.2f}]")
        print(f"  # Material 2 (bone): beta2={beta2:.2f}, alpha2={alpha2:.2f}")
        print(f"     -> covers range [{beta2:.2f}, {beta2+alpha2:.2f}] (max={max_val:.2f})")
        
    else:
        # Single peak or no clear separation
        print(f"\nSingle material distribution detected.")
        print(f"Use 1M-NeAS parameters with beta fixed at 0.1 and alpha covering the full range.")
        print(f"Consider using 1M-NeAS instead of 2M-NeAS")
        print(f"or check if bone regions are present in your data.")
        
        # Provide basic 1M suggestion
        max_val = values_nonzero.max()
        beta = 0.01
        alpha = max_val - beta if max_val > beta else 0.0
        print(f"\nSuggested 1M parameters:")
        print(f"  beta: {beta:.2f}")
        print(f"  alpha: {alpha:.2f}  (covers [{beta:.2f}, {beta+alpha:.2f}])")
        
        # Fallback 2M hints if forced
        val_25 = np.percentile(values_nonzero, 25)
        val_75 = np.percentile(values_nonzero, 75)
        val_max = values_nonzero.max()
        print(f"\nFallback suggestions (if forcing 2M-NeAS):")
        print(f"  # using threshold at 75th percentile")
        print(f"  alpha1: {val_75 - beta:.2f}")
        print(f"  beta1: {beta:.2f}")
        print(f"  beta2: {beta + (val_75 - beta):.2f}")
        print(f"  alpha2: {val_max - (beta + (val_75 - beta)):.2f}")
        
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16)
    
    # Histogram of all values
    axes[0, 0].hist(values, bins=100, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Attenuation Coefficient')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution (All Voxels)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Histogram of non-zero values with peaks
    axes[0, 1].hist(values_nonzero, bins=100, alpha=0.7, edgecolor='black')
    # Mark detected peaks
    for peak_idx in peaks:
        peak_val = bin_centers[peak_idx]
        axes[0, 1].axvline(peak_val, color='red', linestyle='--', linewidth=2, 
                          label=f'Peak: {peak_val:.3f}')
    axes[0, 1].set_xlabel('Attenuation Coefficient')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution (Non-zero Voxels)')
    axes[0, 1].grid(True, alpha=0.3)
    if len(peaks) > 0:
        axes[0, 1].legend()
    
    # Cumulative distribution
    sorted_vals = np.sort(values_nonzero)
    cumulative = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    axes[1, 0].plot(sorted_vals, cumulative, linewidth=2)
    axes[1, 0].set_xlabel('Attenuation Coefficient')
    axes[1, 0].set_ylabel('Cumulative Probability')
    axes[1, 0].set_title('Cumulative Distribution (Non-zero)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 2D slice through volume center
    z_mid = volume.shape[2] // 2
    slice_img = volume[:, :, z_mid]
    im = axes[1, 1].imshow(slice_img, cmap='gray', interpolation='nearest')
    axes[1, 1].set_title(f'Central Slice (z={z_mid})')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], label='Attenuation')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_path}")
    else:
        plt.show()
    
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze attenuation distribution from 1M-NeAS checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to 1M-NeAS checkpoint (e.g., checkpoints/foot_50_1m_hash/checkpoint_epoch_500.pth)')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file (e.g., config/foot_configs/foot_50_1m_hash.yaml)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for visualization (default: checkpoint_dir/attenuation_analysis.png)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--chunk_size', type=int, default=8192,
                       help='Chunk size for processing voxels')
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output is None:
        checkpoint_dir = Path(args.checkpoint).parent
        args.output = str(checkpoint_dir / 'attenuation_analysis.png')
    
    print("Loading model...")
    sdf_model, att_model, s, cfg = load_model(args.checkpoint, args.config, args.device)
    
    print("Loading dataset...")
    data_path = cfg["exp"]["datadir"]
    eval_dset = TIGREDataset(data_path, n_rays=512, type="val", device=args.device)
    
    print(f"Voxel grid shape: {eval_dset.voxels.shape}")
    
    # Sample volume
    pred_volume = sample_volume(sdf_model, att_model, s, eval_dset.voxels, 
                               chunk_size=args.chunk_size, device=args.device)
    
    # Analyze distribution
    title = f"Attenuation Analysis: {Path(args.checkpoint).parent.name}"
    analyze_distribution(pred_volume, output_path=args.output, title=title)
    
    print(f"Analysis complete! Review the suggestions above and update your 2M config.")


if __name__ == "__main__":
    main()
