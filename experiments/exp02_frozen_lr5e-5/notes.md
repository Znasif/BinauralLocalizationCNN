# Exp 02 — All layers, lr=5e-5, 20 epochs (new shuffled tfrecords)

## Setup
- All layers trained (no freezing)
- lr=5e-5, batch=16, epochs=20
- No weight decay, no LR schedule
- New shuffled tfrecords with better train/val distribution (13 shards each for 0-click and 1-click)
- Code: original finetune_custom.py (training_state=True hardcoded, no Az(FB) metric)

## Results (epoch 20)

| Metric | Train | Val |
|---|---|---|
| acc_0click | 52.23% | 52.15% |
| acc_1click | 2.67% | 1.54% |
| acc_original | — | 20.31% |
| loss | 3.16 | 3.27 |
| MAE Az | 23.3° | 23.0° |
| MAE El | 6.3° | 6.5° |

## Observations
- Train/val gap nearly eliminated for 0-click (~52% both) — huge improvement over exp01's 2.6x gap
- New shuffled tfrecords fixed the distribution issue that caused overfitting in exp01
- Val loss (3.27) very close to train loss (3.16) — good generalization
- 1-click still near-zero: precedence effect prior from pretraining remains dominant
- acc_original 20.3% (down from ~27% pretrained baseline): moderate forgetting
- Note: val metrics measured with dropout ON and BN in training mode (bug fixed in exp03)
