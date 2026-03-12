#!/usr/bin/env python3
"""
Generate ground-truth and predicted projection PNGs for a chosen set of
validation views.

Usage:
    python eval_4_views.py \\
        --checkpoint checkpoints/foot_50_4m_hash/checkpoint_epoch_1000.pth \\
        --gt_pickle  data/foot_50.pickle \\
        --views 1 5 10 25 50

Outputs (saved relative to the checkpoint directory):
    <ckpt_dir>/gt/view_001.png   ...  Ground-truth intensity projections
    <ckpt_dir>/preds/view_001.png ...  Model-predicted intensity projections

View numbers are 1-indexed (matching the dataset's 1..numVal range).
"""

import os
import sys
import argparse

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make sure project root is importable (so we can import sibling modules)
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from src.dataset.tigre import TIGREDataset
from src.render.render import render_image

# reuse the same model-building helper from the evaluation script rather
# than duplicating logic here.  That helper also handles two-material
# checkpoints and returns att_model2 when appropriate.
from src.eval_validation_metrics import _build_models_from_checkpoint

# NOTE: we no longer need to import all of the individual network
# constructors; the helper will create the right models for us.


# ---------------------------------------------------------------------------
# Helpers (mirrors eval_validation_metrics.py so we stay consistent)
# ---------------------------------------------------------------------------

# The functionality that was previously implemented here is now
# handled by `_build_models_from_checkpoint` in
# `src/eval_validation_metrics.py`.  We simply wrap it for clarity.

def _build_models(ckpt, device):
    """Wrapper around shared helper.

    Returns:
        sdf_model, att_model1, att_model2, s_param, num_materials
    """
    return _build_models_from_checkpoint(ckpt, device)


def _normalize(pred, gt):
    """Scale pred to match gt via least-squares: s = (pred·gt)/(pred·pred).
    Only used for whole-run normalisation if needed; not applied per-view
    (matching eval_validation_metrics.py which normalises only the 3D volume).
    """
    dot_pp = (pred * pred).sum()
    if dot_pp > 0:
        return pred * ((pred * gt).sum() / dot_pp)
    return pred


def _save_png(array_2d, path):
    """Save a 2-D numpy float array as a grayscale PNG (no axes/border)."""
    fig, ax = plt.subplots(figsize=(array_2d.shape[1] / 100, array_2d.shape[0] / 100),
                           dpi=100)
    ax.imshow(array_2d, cmap="gray")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Save GT and predicted projection PNGs for selected validation views."
    )
    parser.add_argument(
        "--checkpoint", "-c", required=True,
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "--gt_pickle", "-p", required=True,
        help="Path to TIGRE-format pickle file (must contain a 'val' split)",
    )
    parser.add_argument(
        "--views", "-v", nargs="+", type=int, required=True,
        help="1-indexed validation view numbers to render (e.g. --views 1 5 10 25 50)",
    )
    parser.add_argument(
        "--n_samples", type=int, default=128,
        help="Number of samples along each ray for rendering (default: 128)",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=4096,
        help="Ray batch size for rendering (default: 4096)",
    )
    parser.add_argument(
        "--device", "-d", default=None,
        help="Device to use: 'cuda' or 'cpu' (default: auto)",
    )
    args = parser.parse_args()

    # ---- device ----
    if args.device is not None:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---- output dirs ----
    ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    gt_dir   = os.path.join(ckpt_dir, "gt")
    pred_dir = os.path.join(ckpt_dir, "preds")
    os.makedirs(gt_dir,   exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    print(f"GT   output dir : {gt_dir}")
    print(f"Pred output dir : {pred_dir}")

    # ---- load checkpoint & build models ----
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)

    if ckpt.get("encoding_type", "freq") == "hash" and device == "cpu":
        raise RuntimeError(
            "This checkpoint uses hash encoding which requires CUDA. "
            "Run with --device cuda."
        )

    # helper now returns att_model2 for the two-material case
    sdf_model, att_model, att_model2, s_param, num_materials = _build_models(ckpt, device)
    sdf_model.eval()
    att_model.eval()
    if att_model2 is not None:
        att_model2.eval()
    s_tensor = torch.tensor(s_param, device=device)
    print(f"Model loaded  (encoding={ckpt.get('encoding_type','freq')}, "
          f"K={num_materials}, s={s_param:.2f})")

    # ---- load validation dataset ----
    print(f"Loading validation data: {args.gt_pickle}")
    val_ds = TIGREDataset(args.gt_pickle, n_rays=1024, type="val", device=device)
    n_views = val_ds.n_samples
    print(f"Total validation views in pickle: {n_views}")

    # ---- validate requested view numbers ----
    requested = sorted(set(args.views))
    invalid = [v for v in requested if not (1 <= v <= n_views)]
    if invalid:
        parser.error(
            f"The following view numbers are out of range 1..{n_views}: {invalid}"
        )

    # ---- render loop ----
    with torch.no_grad():
        for view_num in requested:
            idx = view_num - 1  # convert to 0-based index

            # Ground truth: exp(-log_attenuation_proj), matching eval_validation_metrics
            projs = val_ds.projs[idx].to(device)
            gt_proj = torch.exp(-projs).cpu().numpy().T  # [H, W]

            gt_path = os.path.join(gt_dir, f"view_{view_num:03d}.png")
            _save_png(gt_proj, gt_path)

            # Predicted intensity projection
            rays = val_ds.rays[idx].to(device)  # [H, W, 8]
            pred_img = render_image(
                rays, sdf_model, att_model, s_tensor,
                args.n_samples,
                chunk_size=args.chunk_size,
                tau=None,
                att_model2=att_model2  # may be None for single-material checkpoints
            )
            pred_np = pred_img.cpu().numpy().T  # [H, W]

            pred_path = os.path.join(pred_dir, f"view_{view_num:03d}.png")
            _save_png(pred_np, pred_path)

            print(f"  View {view_num:3d}/{n_views}  ->  gt: {gt_path}  |  pred: {pred_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
