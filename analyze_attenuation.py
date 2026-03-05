#!/usr/bin/env python3
"""
Analyze attenuation distribution from a trained 1M-NeAS model.

Determines the optimal number of materials K (via Gaussian-mixture BIC) and
suggests alpha/beta parameters for each material, ready to paste into a
KM-NeAS config file.

Usage:
    python analyze_attenuation.py --checkpoint checkpoints/foot_50_1m_hash/checkpoint_epoch_500.pth \
                                   --config config/foot_configs/foot_50_1m_hash.yaml \
                                   --output attenuation_analysis_foot.png
"""

import argparse
import torch
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend; safe for headless servers
import matplotlib.pyplot as plt
from scipy.stats import norm
from pathlib import Path

from src.dataset import TIGREDataset
from src.network import sdf_freq_mlp, sdf_hash_mlp, att_freq_mlp, att_hash_mlp
from src.render import surface_boundary_function


# ---------------------------------------------------------------------------
# Gaussian Mixture Model — EM fitting + BIC-based K selection
# ---------------------------------------------------------------------------

def _fit_gmm_em(X, K, n_init=5, max_iter=300, tol=1e-8):
    """Fit a K-component 1-D Gaussian mixture to X via EM.

    Returns the best of *n_init* random restarts as
    (means, stds, weights, log_likelihood).
    """
    n = len(X)
    x_std = X.std()
    best_ll = -np.inf
    best_params = None

    rng = np.random.default_rng(42)
    for _ in range(n_init):
        # Initialise means at K spread-out quantiles plus a small jitter
        q = np.linspace(5, 95, K)
        means   = np.array([np.percentile(X, qi) for qi in q])
        means  += rng.standard_normal(K) * x_std * 0.05
        stds    = np.full(K, x_std / max(K, 1))
        weights = np.ones(K) / K

        prev_ll = -np.inf
        for _ in range(max_iter):
            # ---------- E-step (log-domain for numerical stability) ----------
            log_resp = np.column_stack([
                np.log(max(w, 1e-300)) + norm.logpdf(X, m, max(s, 1e-8))
                for w, m, s in zip(weights, means, stds)
            ])  # [N, K]
            log_resp -= log_resp.max(axis=1, keepdims=True)
            resp = np.exp(log_resp)
            resp /= resp.sum(axis=1, keepdims=True)

            # ---------- M-step ----------
            Nk      = resp.sum(axis=0)  # [K]
            means   = (resp * X[:, None]).sum(axis=0) / np.maximum(Nk, 1e-8)
            stds    = np.sqrt(
                (resp * (X[:, None] - means) ** 2).sum(axis=0) / np.maximum(Nk, 1e-8)
            )
            stds    = np.maximum(stds, 1e-6)
            weights = Nk / n

            # ---------- Log-likelihood (log-sum-exp trick) ----------
            log_dens = np.column_stack([
                np.log(max(w, 1e-300)) + norm.logpdf(X, m, s)
                for w, m, s in zip(weights, means, stds)
            ])
            lse_max = log_dens.max(axis=1)
            ll = float(
                np.sum(
                    np.log(np.exp(log_dens - lse_max[:, None]).sum(axis=1)) + lse_max
                )
            )
            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

        if ll > best_ll:
            best_ll = ll
            order   = np.argsort(means)
            best_params = (
                means[order].copy(),
                stds[order].copy(),
                weights[order].copy(),
                ll,
            )

    return best_params


def fit_optimal_k(values, K_max=4):
    """Fit GMMs for K=1..K_max and select the best K by BIC.

    Returns:
        optimal_k   : int
        results     : dict  {K: {means, stds, weights, ll, bic, aic}}
    """
    results = {}
    n = len(values)
    for K in range(1, K_max + 1):
        means, stds, weights, ll = _fit_gmm_em(values, K)
        n_params = 3 * K - 1          # K means + K stds + (K-1) free weights
        bic = -2 * ll + n_params * np.log(n)
        aic = -2 * ll + 2 * n_params
        results[K] = dict(means=means, stds=stds, weights=weights,
                          ll=ll, bic=bic, aic=aic)

    optimal_k = min(results, key=lambda k: results[k]['bic'])
    return optimal_k, results


def compute_material_configs(means, stds, values):
    """Derive (alpha, beta) pairs from K sorted Gaussian components.

    Material boundaries are placed at the density valley between adjacent
    peaks.  The leftmost boundary is always 0; the rightmost is 15 % above
    the 99.5th-percentile of *values*.

    Returns:
        material_configs : list of (alpha, beta) tuples
        boundaries       : list of K+1 boundary values
    """
    K = len(means)
    boundaries = [0.0]

    for i in range(K - 1):
        mu1, s1 = means[i],     stds[i]
        mu2, s2 = means[i + 1], stds[i + 1]
        # Find the local density minimum between the two peaks
        x_search = np.linspace(mu1, mu2, 500)
        density  = norm.pdf(x_search, mu1, s1) + norm.pdf(x_search, mu2, s2)
        boundaries.append(float(x_search[np.argmin(density)]))

    right_bound = float(np.percentile(values, 99.5) * 1.15)
    boundaries.append(right_bound)

    material_configs = []
    for i in range(K):
        lo    = boundaries[i]
        hi    = boundaries[i + 1]
        beta  = round(max(0.0, lo), 4)
        alpha = round(hi - lo, 4)
        material_configs.append((alpha, beta))

    return material_configs, boundaries



def load_model(checkpoint_path, config_path, device='cuda'):
    """Load a trained 1M-NeAS model from a checkpoint."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    num_materials = cfg["network"].get("num_materials", 1)
    if num_materials != 1:
        raise ValueError(
            f"This script requires a 1M-NeAS checkpoint, "
            f"but config has num_materials={num_materials}"
        )

    ckpt = torch.load(checkpoint_path, map_location=device)

    feature_dim   = cfg["network"].get("feature_dim", 8)
    multires      = cfg["network"].get("multires", 6)
    encoding_type = cfg["network"].get("encoding_type", "freq")

    # Read material parameters from the unified list format
    mats  = cfg["network"].get("materials", [{}])
    alpha = mats[0].get("alpha", 3.4)
    beta  = mats[0].get("beta",  0.1)

    if encoding_type == "freq":
        sdf_model = sdf_freq_mlp(
            input_dim=3, output_dim=1, feature_dim=feature_dim, multires=multires
        ).to(device)
        att_model = att_freq_mlp(
            input_dim=feature_dim, output_dim=1, alpha=alpha, beta=beta
        ).to(device)
    elif encoding_type == "hash":
        num_levels        = cfg["network"].get("num_levels", 14)
        level_dim         = cfg["network"].get("level_dim", 2)
        base_resolution   = cfg["network"].get("base_resolution", 16)
        log2_hashmap_size = cfg["network"].get("log2_hashmap_size", 19)
        sdf_model = sdf_hash_mlp(
            input_dim=3, output_dim=1, feature_dim=feature_dim,
            num_levels=num_levels, level_dim=level_dim,
            base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size,
        ).to(device)
        att_model = att_hash_mlp(
            input_dim=feature_dim, output_dim=1,
            num_levels=num_levels, level_dim=level_dim,
            base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size,
            alpha=alpha, beta=beta,
        ).to(device)
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")

    sdf_model.load_state_dict(ckpt["sdf_model_state_dict"])
    att_state = ckpt.get("att_model1_state_dict", ckpt.get("att_model_state_dict"))
    if att_state is not None:
        att_model.load_state_dict(att_state)

    sdf_model.eval()
    att_model.eval()

    return sdf_model, att_model, ckpt["s"], cfg


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


# ---------------------------------------------------------------------------
# Distribution analysis
# ---------------------------------------------------------------------------

MATERIAL_NAMES = [
    "air/background",
    "soft tissue",
    "dense tissue / cartilage",
    "bone / cortical",
    "metal / implant",
]


def analyze_distribution(volume, output_path=None, title="Attenuation Distribution",
                         air_threshold=0.01, K_max=4):
    """Fit a Gaussian mixture to the non-background voxels, determine optimal K
    by BIC, and print/plot the resulting alpha/beta suggestions."""

    values = volume.flatten()
    values_nonzero = values[values > air_threshold]

    # ---- Basic statistics ----
    print(f"\n{'='*60}")
    print("ATTENUATION DISTRIBUTION ANALYSIS")
    print(f"{'='*60}")
    print(f"Total voxels         : {len(values):,}")
    print(f"Non-background (>{air_threshold:.3f}): {len(values_nonzero):,}")
    print(f"Overall range        : [{values.min():.4f}, {values.max():.4f}]")
    print(f"Non-bg range         : [{values_nonzero.min():.4f}, {values_nonzero.max():.4f}]")
    print(f"Mean  (non-bg)       : {values_nonzero.mean():.4f}")
    print(f"Median (non-bg)      : {np.median(values_nonzero):.4f}")
    print(f"Std   (non-bg)       : {values_nonzero.std():.4f}")
    print()
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  {p:2d}th percentile: {np.percentile(values_nonzero, p):.4f}")

    # ---- Subsample for speed (GMM EM on large arrays is slow) ----
    MAX_FIT = 50_000
    if len(values_nonzero) > MAX_FIT:
        rng = np.random.default_rng(0)
        fit_vals = rng.choice(values_nonzero, size=MAX_FIT, replace=False)
    else:
        fit_vals = values_nonzero

    # ---- GMM fitting for K=1..K_max ----
    print(f"\nFitting Gaussian mixtures (K=1..{K_max}) …")
    optimal_k, gmm_results = fit_optimal_k(fit_vals, K_max=K_max)

    print(f"\n{'='*60}")
    print("BIC SCORES (lower = better)")
    print(f"{'='*60}")
    for K, res in gmm_results.items():
        marker = "  <-- OPTIMAL" if K == optimal_k else ""
        print(f"  K={K}: BIC = {res['bic']:12.2f}  AIC = {res['aic']:12.2f}{marker}")

    # ---- Compute alpha/beta from optimal GMM ----
    best = gmm_results[optimal_k]
    material_configs, boundaries = compute_material_configs(
        best['means'], best['stds'], values_nonzero
    )

    # ---- Print results ----
    print(f"\n{'='*60}")
    print(f"OPTIMAL K = {optimal_k}")
    print(f"{'='*60}")
    for i, ((alpha, beta), mu, sigma, w) in enumerate(
        zip(material_configs, best['means'], best['stds'], best['weights'])
    ):
        label = MATERIAL_NAMES[i + 1] if (i + 1) < len(MATERIAL_NAMES) else f"material {i+1}"
        lo, hi = boundaries[i], boundaries[i + 1]
        print(
            f"  Material {i+1} ({label}): "
            f"peak={mu:.4f}  σ={sigma:.4f}  weight={w:.3f}  "
            f"range=[{lo:.4f}, {hi:.4f}]"
        )
        print(f"    alpha: {alpha}   beta: {beta}")

    # ---- YAML snippet ----
    print(f"\n{'='*60}")
    print("CONFIG YAML SNIPPET (paste into your KM-NeAS config)")
    print(f"{'='*60}")
    print(f"  num_materials: {optimal_k}")
    print( "  materials:")
    for alpha, beta in material_configs:
        print(f"    - {{alpha: {alpha}, beta: {beta}}}")
    print()

    # Also show suggestions for all K values
    print(f"{'='*60}")
    print("SUGGESTIONS FOR ALL K VALUES")
    print(f"{'='*60}")
    for K, res in gmm_results.items():
        mat_cfgs, _ = compute_material_configs(res['means'], res['stds'], values_nonzero)
        marker = "  <-- BIC-optimal" if K == optimal_k else ""
        print(f"\n  K={K}{marker}")
        print(f"    num_materials: {K}")
        print( "    materials:")
        for alpha, beta in mat_cfgs:
            print(f"      - {{alpha: {alpha}, beta: {beta}}}")
    print()

    # ---- Visualization ----
    COLORS = plt.cm.tab10.colors  # type: ignore[attr-defined]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14)

    # Panel 1 — full histogram
    axes[0, 0].hist(values, bins=150, alpha=0.7, color='steelblue', edgecolor='none')
    axes[0, 0].axvline(air_threshold, color='red', linestyle='--',
                       linewidth=1.2, label=f'air threshold ({air_threshold})')
    axes[0, 0].set_xlabel('Attenuation coefficient')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Full distribution')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # Panel 2 — non-background histogram + fitted GMM components + boundaries
    axes[0, 1].hist(values_nonzero, bins=150, density=True,
                    alpha=0.4, color='steelblue', edgecolor='none', label='data')
    x_plot = np.linspace(values_nonzero.min(), values_nonzero.max(), 500)
    total_density = np.zeros_like(x_plot)
    for i, (mu, sigma, w) in enumerate(
        zip(best['means'], best['stds'], best['weights'])
    ):
        comp = w * norm.pdf(x_plot, mu, sigma)
        total_density += comp
        axes[0, 1].fill_between(x_plot, comp, alpha=0.35, color=COLORS[i],
                                label=f'M{i+1}: μ={mu:.3f}')
    axes[0, 1].plot(x_plot, total_density, 'k-', linewidth=1.5, label='mixture')
    for j, bnd in enumerate(boundaries[1:-1]):
        axes[0, 1].axvline(bnd, color='black', linestyle=':', linewidth=1.2,
                           label=f'boundary {j+1}→{j+2}: {bnd:.3f}')
    axes[0, 1].set_xlabel('Attenuation coefficient')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title(f'GMM fit  (optimal K={optimal_k})')
    axes[0, 1].legend(fontsize=7)
    axes[0, 1].grid(True, alpha=0.3)

    # Panel 3 — BIC / AIC vs K
    ks   = list(gmm_results.keys())
    bics = [gmm_results[k]['bic'] for k in ks]
    aics = [gmm_results[k]['aic'] for k in ks]
    x_pos = np.arange(len(ks))
    width = 0.35
    axes[1, 0].bar(x_pos - width/2, bics, width, label='BIC', alpha=0.8, color='steelblue')
    axes[1, 0].bar(x_pos + width/2, aics, width, label='AIC', alpha=0.8, color='darkorange')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels([f'K={k}' for k in ks])
    axes[1, 0].set_ylabel('Score (lower = better)')
    axes[1, 0].set_title('Model selection (BIC / AIC)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    # Highlight optimal K
    opt_idx = ks.index(optimal_k)
    axes[1, 0].get_xticklabels()[opt_idx].set_fontweight('bold')
    axes[1, 0].get_xticklabels()[opt_idx].set_color('red')

    # Panel 4 — central volume slice
    z_mid = volume.shape[2] // 2
    im = axes[1, 1].imshow(volume[:, :, z_mid], cmap='gray', interpolation='nearest')
    axes[1, 1].set_title(f'Central axial slice (z={z_mid})')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], label='Attenuation')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    else:
        plt.show()

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze attenuation distribution from a 1M-NeAS checkpoint '
                    'and suggest optimal K + alpha/beta config.'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to 1M-NeAS checkpoint (.pth)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to matching config YAML')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for visualization PNG '
                             '(default: <checkpoint_dir>/attenuation_analysis.png)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--chunk_size', type=int, default=8192,
                        help='Chunk size for processing voxels')
    parser.add_argument('--k_max', type=int, default=4,
                        help='Maximum number of materials to consider (default: 4)')
    parser.add_argument('--air_threshold', type=float, default=0.01,
                        help='Attenuation threshold below which voxels are '
                             'treated as air/background (default: 0.01)')

    args = parser.parse_args()

    if args.output is None:
        checkpoint_dir = Path(args.checkpoint).parent
        args.output = str(checkpoint_dir / 'attenuation_analysis.png')

    print("Loading model...")
    sdf_model, att_model, s, cfg = load_model(args.checkpoint, args.config, args.device)

    print("Loading dataset...")
    data_path = cfg["exp"]["datadir"]
    eval_dset = TIGREDataset(data_path, n_rays=512, type="val", device=args.device)

    print(f"Voxel grid shape: {eval_dset.voxels.shape}")

    pred_volume = sample_volume(
        sdf_model, att_model, s, eval_dset.voxels,
        chunk_size=args.chunk_size, device=args.device,
    )

    title = f"Attenuation Analysis: {Path(args.checkpoint).parent.name}"
    analyze_distribution(
        pred_volume,
        output_path=args.output,
        title=title,
        air_threshold=args.air_threshold,
        K_max=args.k_max,
    )

    print("Analysis complete. Copy the YAML snippet above into your KM-NeAS config.")


if __name__ == "__main__":
    main()
