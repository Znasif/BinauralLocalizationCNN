# Exp 02 — ABORTED at epoch 3

## Setup
- freeze_conv=True, lr=1e-5, weight_decay=1e-4, epochs=30

## What happened
Aborted. Loss stuck at ~48 (vs exp01's ~3-5). Root cause: L2 weight decay was being
computed over ALL trainable variables including the frozen conv layers. With millions
of large pretrained conv weights, the L2 term (~42 loss units) completely drowned
the cross-entropy signal (~6.2 expected for 504 classes at random).

## Fix applied
- `finetune_custom.py` now computes L2 only on the vars actually being trained
  (i.e., scoped to `var_list` when freeze_conv is active)
- Updated run.sh to remove `--weight_decay` (freeze alone is sufficient)

## What to try next (exp03)
- Same as intended exp02 but WITHOUT weight_decay bug
- See experiments/exp03_frozen_lr1e-5/
