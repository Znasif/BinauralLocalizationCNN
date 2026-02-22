#!/bin/bash
# Experiment 02 — Frozen conv layers, lr=1e-5
# NOTE: weight_decay was dropped — L2 on FC-only vars is negligible and adds instability
# Use exp03 for the clean frozen run (this run was aborted at epoch 3 due to loss ~48)
# Root cause: weight_decay was being applied to all vars including frozen conv weights

EXP_DIR="experiments/exp02_frozen_lr1e-5"

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

