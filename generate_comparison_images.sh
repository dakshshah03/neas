#!/usr/bin/env bash
set -euo pipefail

# Generate GT and predicted projection PNGs for selected validation views
# across all 4 anatomies and all model sizes, using eval_4_views.py.
#
# Outputs are saved under:
#   checkpoints/<anatomy>_50_<size>_hash/gt/view_NNN.png
#   checkpoints/<anatomy>_50_<size>_hash/preds/view_NNN.png
#
# Usage:
#   ./generate_comparison_images.sh [--gpu DEVICE] [--views "1 5 10 25 50"] [--epoch N]
#
# Examples:
#   ./generate_comparison_images.sh
#   ./generate_comparison_images.sh --gpu cuda:1 --views "1 10 25 50" --epoch 500

# ---- defaults ---------------------------------------------------------------
GPU_DEVICE="cuda"
EPOCH=1000
VIEWS="1 15 30 45"

# ---- parse CLI --------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu|-g)
      GPU_DEVICE="$2"; shift 2 ;;
    --epoch|-e)
      EPOCH="$2"; shift 2 ;;
    --views|-v)
      VIEWS="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--gpu DEVICE] [--views \"1 5 10 25 50\"] [--epoch N]"
      exit 1 ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY_CMD="python"
ANATOMIES=(chest foot jaw abdomen)
SIZES=(1m 2m)
VARIANT="hash"

echo "============================================================"
echo "  generate_comparison_images.sh"
echo "  epoch   : ${EPOCH}"
echo "  views   : ${VIEWS}"
echo "  device  : ${GPU_DEVICE}"
echo "============================================================"
echo

for a in "${ANATOMIES[@]}"; do
  for s in "${SIZES[@]}"; do
    CKPT_PATH="$ROOT_DIR/checkpoints/${a}_50_${s}_${VARIANT}/checkpoint_epoch_${EPOCH}.pth"
    VAL_PICKLE="$ROOT_DIR/data/${a}_50.pickle"

    echo "------------------------------------------------------------"
    echo "anatomy=${a}  size=${s}  epoch=${EPOCH}  variant=${VARIANT}"
    echo "Checkpoint : ${CKPT_PATH}"
    echo "Val pickle : ${VAL_PICKLE}"

    if [[ ! -f "$CKPT_PATH" ]]; then
      echo "⚠️  Checkpoint not found — skipping: $CKPT_PATH" >&2
      continue
    fi
    if [[ ! -f "$VAL_PICKLE" ]]; then
      echo "⚠️  Validation pickle not found — skipping: $VAL_PICKLE" >&2
      continue
    fi

    # shellcheck disable=SC2086
    "$PY_CMD" "$ROOT_DIR/eval_4_views.py" \
      --checkpoint "$CKPT_PATH" \
      --gt_pickle  "$VAL_PICKLE" \
      --views $VIEWS \
      --device "$GPU_DEVICE"

    echo "✅  Done — images saved under $(dirname "$CKPT_PATH")/gt  and  $(dirname "$CKPT_PATH")/preds"
    echo
  done
done

echo "============================================================"
echo "✅  All done."
