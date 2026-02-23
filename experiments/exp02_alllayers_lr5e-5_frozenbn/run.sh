#!/bin/bash
# Experiment 02 — All layers, lr=1e-5, frozen BN stats, 10 epochs
# Goal: preserve ORIG AzFB near 6° by limiting weight drift
# Lower LR (1e-5) + fewer epochs (10) + frozen BN stats

EXP_DIR="experiments/exp02_alllayers_lr5e-5_frozenbn"

tensorboard --logdir "${EXP_DIR}/logs/" --port 6006 &
TB_PID=$!
echo "TensorBoard started (PID $TB_PID) → http://localhost:6006"
trap "echo 'Stopping...'; kill $TB_PID; exit" INT TERM

python finetune_custom.py \
  --tfrecords_dir ./echo_finetune_tfrecords/ \
  --model_dir models/net1 \
  --output_dir "${EXP_DIR}/checkpoints" \
  --log_dir "${EXP_DIR}/logs" \
  --epochs 10 \
  --batch_size 16 \
  --lr 1e-5 \
  --freeze_bn_stats \
  --save_best_val

kill $TB_PID
