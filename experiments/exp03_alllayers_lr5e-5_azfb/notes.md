# Exp 03 — All layers, lr=5e-5, 20 epochs (with Az(FB) + is_training fix)

## Setup
- All layers trained (no freezing)
- lr=5e-5, batch=16, epochs=20
- No weight decay, no LR schedule
- Same shuffled tfrecords as exp02
- Code fixes applied: is_training placeholder (dropout OFF + BN moving avgs during val), Az(FB) metric

## Results (epoch 20)

| Metric | Train | Val |
|---|---|---|
| acc_0click | 52.50% | 62.91% |
| acc_1click | 2.53% | 2.56% |
| acc_original | — | 4.17% |
| loss | 3.20 | 2.92 |
| MAE Az | 22.8° | 18.8° |
| MAE Az(FB) | 12.3° | 10.5° |
| MAE El | 6.3° | 5.8° |

Paper baseline (pretrained, original data): Az(FB) 4.4°, El 5.3°

## Observations
- Val acc_0click 62.9% vs exp02's 52.2%: the ~10pp boost is from disabling dropout during val (is_training fix)
- Val loss (2.92) now **lower** than train loss (3.20) — expected since train has dropout active, val does not
- Az(FB) 10.5° on val: ~2.4x the paper's 4.4° baseline, but on harder precedence-effect stimuli not original data
- El 5.8° on val: close to paper's 5.3° — elevation localization transfers well
- 1-click still ~2.5%: precedence effect prior remains dominant
- **acc_original collapsed to 4.2%** (vs exp02's 20.3%): severe catastrophic forgetting
  - With is_training=False (BN using moving avgs), the BN stats that drifted during finetuning now fully expose how much the model has diverged from pretrained distribution
  - exp02 showed 20.3% but that was with dropout ON masking the true extent of forgetting
- Best val checkpoint: acc_0click=62.91% at epoch 20

## Comparison: exp02 vs exp03

| Metric | exp02 val | exp03 val | Delta | Cause |
|---|---|---|---|---|
| acc_0click | 52.15% | 62.91% | +10.8pp | is_training fix (no dropout during val) |
| loss | 3.27 | 2.92 | -0.35 | is_training fix (no dropout during val) |
| acc_original | 20.31% | 4.17% | -16.1pp | BN moving avgs expose true forgetting |
| MAE Az | 23.0° | 18.8° | -4.2° | is_training fix |
| MAE El | 6.5° | 5.8° | -0.7° | is_training fix |

Key takeaway: exp02 and exp03 trained identically (same model weights). The difference is purely in **how validation is measured**. exp03's numbers are the correct inference-time metrics.
