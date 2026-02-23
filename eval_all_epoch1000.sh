#!/usr/bin/env bash
set -euo pipefail

# Run evaluation for 1m and 2m models for all 4 anatomies (uses checkpoint_epoch_1000)
# The underlying Python script now also writes example projection slices and a
# few 3D volume slices under eval/<yaml_id>. By default this root folder is the
# "eval" subdirectory of the checkpoint being evaluated, so outputs appear
# alongside the checkpoint's own eval/ directory structure.  You can still
# override via the EVAL_ROOT environment variable if needed.
#
# Usage: ./eval_all_epoch1000.sh [--gpu DEVICE]
# Example: ./eval_all_epoch1000.sh --gpu cuda:0

# Parse CLI arguments
GPU_DEVICE="cuda"
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu|-g)
      GPU_DEVICE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--gpu DEVICE]"
      exit 1
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY_CMD="python"
ANATOMIES=(chest foot jaw abdomen)
SIZES=(1m 2m)
VARIANT="hash"
EPOCH=1000

# Summary CSV (one row per evaluated config/yaml)
# - You can override OUTPUT_CSV or YAML_ID_STYLE by exporting them in your shell
OUTPUT_CSV="${OUTPUT_CSV:-$ROOT_DIR/eval_summary_epoch_${EPOCH}.csv}"
# YAML identifier style: anatomy_size_variant | checkpoint_basename | val_pickle
YAML_ID_STYLE="anatomy_size_variant"  # recommended
# CSV mode: overwrite (default) or append
CSV_MODE="overwrite"

# Write header if overwriting or the file doesn't exist
if [[ "$CSV_MODE" == "overwrite" || ! -f "$OUTPUT_CSV" ]]; then
  printf "%s\n" "yaml,anatomy,size,variant,epoch,avg_proj_psnr,avg_proj_ssim,vol_psnr,vol_ssim" > "$OUTPUT_CSV"
fi

for a in "${ANATOMIES[@]}"; do
  for s in "${SIZES[@]}"; do
    CKPT_PATH="$ROOT_DIR/checkpoints/${a}_50_${s}_${VARIANT}/checkpoint_epoch_${EPOCH}.pth"
    VAL_PICKLE="$ROOT_DIR/data/${a}_50.pickle"

    echo "------------------------------------------------------------"
    echo "Running: anatomy=${a}, size=${s}, epoch=${EPOCH}, variant=${VARIANT}"
    echo "Checkpoint: ${CKPT_PATH}"
    echo "Val pickle: ${VAL_PICKLE}"

    if [[ ! -f "$CKPT_PATH" ]]; then
      echo "⚠️  Checkpoint not found — skipping: $CKPT_PATH" >&2
      continue
    fi
    if [[ ! -f "$VAL_PICKLE" ]]; then
      echo "⚠️  Validation pickle not found — skipping: $VAL_PICKLE" >&2
      continue
    fi

    TMP_CSV="$ROOT_DIR/.tmp_eval_${a}_${s}_${VARIANT}_epoch${EPOCH}.csv"

    case "$YAML_ID_STYLE" in
      anatomy_size_variant) YAML_ID="${a}_${s}_${VARIANT}" ;;
      checkpoint_basename) YAML_ID="$(basename "$CKPT_PATH")" ;;
      val_pickle) YAML_ID="$(basename "$VAL_PICKLE")" ;;
      *) YAML_ID="${a}_${s}_${VARIANT}" ;;
    esac

    # set evaluation output root inside checkpoint directory
    CKPT_DIR="$(dirname "$CKPT_PATH")"
    # prefer explicit environment override, otherwise use checkpoint-specific eval folder
    if [[ -n "${EVAL_ROOT:-}" ]]; then
        TARGET_EVAL_ROOT="$EVAL_ROOT"
    else
        TARGET_EVAL_ROOT="$CKPT_DIR/eval"
    fi
    mkdir -p "$TARGET_EVAL_ROOT"

    "$PY_CMD" src/eval_validation_metrics.py \
      --checkpoint "$CKPT_PATH" \
      --val_pickle "$VAL_PICKLE" \
      --device "$GPU_DEVICE" \
      --save_csv "$TMP_CSV" \
      --eval_id "$YAML_ID" \
      --eval_root "$TARGET_EVAL_ROOT"

    # parse avg (projection) and vol (3D) from the per-run CSV (header: view,proj_psnr,proj_ssim)
    if [[ -f "$TMP_CSV" ]]; then
      read -r AVG_PROJ_PSNR AVG_PROJ_SSIM < <(awk -F, '$1=="avg"{gsub(/^[ \t]+|[ \t]+$/, "", $2); gsub(/^[ \t]+|[ \t]+$/, "", $3); print $2, $3; exit}' "$TMP_CSV")
      read -r VOL_PSNR VOL_SSIM       < <(awk -F, '$1=="vol"{gsub(/^[ \t]+|[ \t]+$/, "", $2); gsub(/^[ \t]+|[ \t]+$/, "", $3); print $2, $3; exit}' "$TMP_CSV")
      rm -f "$TMP_CSV"
    else
      echo "⚠️  Expected per-run CSV not found: $TMP_CSV" >&2
      continue
    fi


    # append summary row: yaml,anatomy,size,variant,epoch,avg_proj_psnr,avg_proj_ssim,vol_psnr,vol_ssim
    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
      "$YAML_ID" "$a" "$s" "$VARIANT" "$EPOCH" "$AVG_PROJ_PSNR" "$AVG_PROJ_SSIM" "$VOL_PSNR" "$VOL_SSIM" >> "$OUTPUT_CSV"

    echo "✅ Metrics appended to $OUTPUT_CSV"
    echo
  done
done

echo "✅ All done. (used checkpoint_epoch_${EPOCH} for all runs)"
echo "Summary CSV: $OUTPUT_CSV"