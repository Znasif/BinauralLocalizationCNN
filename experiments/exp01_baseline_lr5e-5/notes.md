# Exp 01 — Baseline: full-network finetune, lr=5e-5, 20 epochs

## Setup
- All layers trained (no freezing)
- lr=5e-5, batch=16, epochs=20
- No weight decay, no LR schedule
- Train/val split: deterministic (sorted shards, last 20% as val)

## Results (epoch 20)

| Metric | Train | Val |
|---|---|---|
| acc_0click | 50.68% | 19.57% |
| acc_1click | 2.98% | 0.00% |
| acc_original | — | 23.44% |
| loss | 3.22 | 5.93 |

## Observations
- Large train/val gap on 0-click (2.6×) → pure overfitting
- 1-click val=0% expected: model fighting the precedence effect prior baked in by 1.6M reverberant training examples
- acc_original drifted from ~26% → ~23.4% over 20 epochs: mild but real forgetting
- Val acc_0click plateaued ~epoch 14-15, then oscillated; best val was ~21% around epoch 15