# NeAS: Neural Attenuation Fields
Unofficial implementation of [NeAS](https://arxiv.org/abs/2503.07491)
```Zhu, Chengrui, et al. "NeAS: 3D Reconstruction from X-ray Images using Neural Attenuation Surface." arXiv preprint arXiv:2503.07491 (2025).```
that works with TIGRE formatted data.
This is still in progress, currently only having implemented frequency encoding and single material.


## Prerequisites

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. Ensure you have it installed.

## Data Preparation

1. Download the dataset from [this Google Drive link](https://drive.google.com/drive/folders/1BJYR4a4iHpfFFOAdbEe5O_7Itt1nukJd).
2. Create a `data` directory in the root of the project if it doesn't exist.
3. Place the downloaded pickle file (e.g., `foot_50.pickle`) inside the `data` directory.
## Setup (uv sync)

1. Install uv (system-wide or in a virtualenv):
```bash
pip install uv
```

2. Sync the project environment / dependencies:
```bash
uv sync
```

This installs the project's Python dependencies into your active environment.

## Quickstart / Usage

All scripts live under `src/`. It's recommended to run them via `uv run` so they execute inside the synced environment.

- Train the model:
```bash
uv run src/train.py --data_path data/foot_50.pickle
```
Key args (examples):
- `--epochs` (default 300)
- `--batch_size` (default 512)
- `--checkpoint_dir` (default ./checkpoints/)
- `--feature_dim`, `--s_param`, etc.

Training saves checkpoints in `checkpoints/<dataset>_<timestamp>/`. Every `--save_interval` epochs a checkpoint is written and validation images are generated under a `val_epoch_<N>/` folder.

- Run validation/compute metrics and save predictions (LPIPS, SSIM, PSNR), and extract mesh:
```bash
uv run src/metrics.py --model_path CHECKPOINT.pth --val_pickle data/foot_50.pickle --save_dir results/val_run
```
Notes:
- If `--save_dir` is omitted, a folder is created next to the checkpoint (e.g. `validation_and_mesh_epoch_<N>`).
- The script:
  - Renders predictions and saves GT and predicted images under `<save_dir>/gt` and `<save_dir>/pred`.
  - Computes LPIPS, SSIM, PSNR and writes `validation_metrics.csv`.
  - Extracts an SDF mesh and writes `mesh.ply` in the same `save_dir`.
- `--device` can be set (e.g., `cuda` or `cpu`). By default it auto-selects `cuda` if available.

- Extract mesh only (standalone):
```bash
uv run src/mesh.py CHECKPOINT.pth --feature_dim 8 --resolution 256 --out out_mesh.ply
```
This runs marching cubes on the trained SDF and writes a PLY file.

## File Overview

- `src/train.py` - train loop
- `src/metrics.py` — Runs validation rendering from a checkpoint, computes & saves LPIPS/SSIM/PSNR, saves predictions and extracts mesh.
- `src/mesh.py` — Utility and CLI to extract mesh from an SDF checkpoint (uses scikit-image marching_cubes).
- `src/mlp.py`, `src/dataset.py` — Model definitions and TIGRE dataset loader used by the above scripts.

## Notes

- The code auto-selects GPU (`cuda`) if available. To force CPU set `--device cpu` when running scripts.
- Output folders and filenames are printed when scripts run; check those locations for results and logs.
