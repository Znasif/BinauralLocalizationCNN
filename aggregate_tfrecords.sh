#!/bin/bash
# Helper script to run aggregate evaluation across all TFRecords in echo_finetune_tfrecords

# Default parameters (can be overridden but provides a sane default)
MODEL_DIR="models/net1"
CHECKPOINT="experiments/exp02_alllayers_lr5e-5_frozenbn/checkpoints/best_val.ckpt"
TFRECORDS_DIR="echo_finetune_tfrecords"
OUT_PLOT="./aggregate_plots/echo_finetune_aggregate_eval1.png"

# Display configuration
SHOW_RAW_OUTPUT=false

echo "============================================================"
echo "Evaluating Finetuning Performance on: $TFRECORDS_DIR"
echo "Model Dir : $MODEL_DIR"
echo "Checkpoint: $CHECKPOINT"
echo "Output    : $OUT_PLOT"
echo "============================================================"

# Create output dir if needed
mkdir -p $(dirname "$OUT_PLOT")

if [ "$SHOW_RAW_OUTPUT" = true ]; then
  # Run normally with full python output
  python test_inference_minimal.py \
    --model_dir "$MODEL_DIR" \
    --checkpoint "$CHECKPOINT" \
    --tfrecords_dir "$TFRECORDS_DIR" \
    --aggregate "$OUT_PLOT"
else
  echo -e "\nRunning Inference (Quiet Mode)..."
  
  # Helper to draw a progress bar
  draw_progress_bar() {
      local _current=$1
      local _total=$2
      local _bar_size=40
      local _completed=$((_current * _bar_size / _total))
      local _remaining=$((_bar_size - _completed))
      local _percent=$((_current * 100 / _total))
      
      printf "\rProgress: ["
      printf "%${_completed}s" | tr ' ' '#'
      printf "%${_remaining}s" | tr ' ' '-'
      printf "] %d%% (%d/%d)" "$_percent" "$_current" "$_total"
  }

  # Run quietly and intercept PROGRESS lines
  python -u test_inference_minimal.py \
    --model_dir "$MODEL_DIR" \
    --checkpoint "$CHECKPOINT" \
    --tfrecords_dir "$TFRECORDS_DIR" \
    --aggregate "$OUT_PLOT" \
    --quiet 2>&1 | while read -r line; do
      if [[ "$line" == PROGRESS:* ]]; then
          IFS='/' read -r current total <<< "${line#PROGRESS: }"
          draw_progress_bar "$current" "$total"
      else
          # Print out normal messages (like errors or initialization) above the progress bar
          # Clear the current line so the progress bar is erased before printing text
          printf "\r\033[K%s\n" "$line"
      fi
  done
  echo "" # Newline after progress bar finishes
fi
