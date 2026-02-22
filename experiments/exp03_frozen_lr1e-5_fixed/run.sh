#!/bin/bash
# Experiment 03 — Frozen conv layers, lr=1e-5, no weight decay
# Fix of exp02 (which was killed due to L2 being applied to frozen conv weights)
# Run from inside Docker: bash experiments/exp03_frozen_lr1e-5_fixed/run.sh

EXP_DIR="experiments/exp03_frozen_lr1e-5_fixed"

tensorboard --logdir "${EXP_DIR}/logs/" --port 6006 &
TB_PID=$!
echo "TensorBoard started (PID $TB_PID) → http://localhost:6006"
trap "echo 'Stopping...'; kill $TB_PID; exit" INT TERM

python finetune_custom.py \
  --tfrecords_dir ./echo_finetune_tfrecords/ \
  --model_dir models/net1 \
  --output_dir "${EXP_DIR}/checkpoints" \
  --log_dir "${EXP_DIR}/logs" \
  --epochs 30 \
  --batch_size 16 \
  --lr 1e-5 \
  --freeze_conv \
  --save_best_val

kill $TB_PID
