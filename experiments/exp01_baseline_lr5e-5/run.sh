#!/bin/bash
# Experiment 01 — Baseline full-network finetune, lr=5e-5, 20 epochs
# Run from inside Docker: bash experiments/exp01_baseline_lr5e-5/run.sh

python finetune_custom.py \
  --tfrecords_dir ./echo_finetune_tfrecords/ \
  --model_dir models/net1 \
  --output_dir experiments/exp01_baseline_lr5e-5/checkpoints \
  --log_dir experiments/exp01_baseline_lr5e-5/logs \
  --epochs 20 \
  --batch_size 16 \
  --lr 5e-5
