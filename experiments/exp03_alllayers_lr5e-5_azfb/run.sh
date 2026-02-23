#!/bin/bash
# Experiment 03 — All layers, lr=5e-5, 20 epochs
# Same as exp02 but with Az(FB) metric and is_training fix for val
# Run from inside Docker: bash experiments/exp03_alllayers_lr5e-5_azfb/run.sh

EXP_DIR="experiments/exp03_alllayers_lr5e-5_azfb"

tensorboard --logdir "${EXP_DIR}/logs/" --port 6006 &
TB_PID=$!
echo "TensorBoard started (PID $TB_PID) → http://localhost:6006"
trap "echo 'Stopping...'; kill $TB_PID; exit" INT TERM

python finetune_custom.py \
  --tfrecords_dir ./echo_finetune_tfrecords/ \
  --model_dir models/net1 \
  --output_dir "${EXP_DIR}/checkpoints" \
  --log_dir "${EXP_DIR}/logs" \
  --epochs 20 \
  --batch_size 16 \
  --lr 5e-5 \
  --save_best_val

kill $TB_PID
