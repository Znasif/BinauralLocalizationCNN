# BinauralLocalizationCNN
**Code to create networks that localize sounds sources in 3D environments**

1. Main training/testing python script is `call_model_training_valid_pad_francl.py`.
    * This script is responsible for processing the experiment parameters, validating the model folder, saving a copy of the experiment parameters there and ensuring the same folder isn't used for two different training rounds.
    * An example set of parameters can be found for testing in `slurm_CNN_spherical_testing_francl.sh` and in `slurm_CNN_spherical_training_francl.sh`.
3. Networks weights can be downloaded at: https://www.dropbox.com/sh/af6vaotxt41i7pe/AACfTzMxMLfv-Edmn33S4gTpa?dl=0

2. The model input should be nervegrams with their associated metadata saved into tensorflow records. The cochlear model we use is the [PyCochleagram package](https://github.com/mcdermottLab/pycochleagram). We have a wrapper to transform stereo `.wav` files into the proper input available here: https://github.com/afrancl/BinauralDataGen
    * The precomputed training data that were used to train the original model can be downloaded here: https://drive.google.com/drive/folders/1pUyMmkurxEWFeofbLKy5-QVWrcMbM8Wz?usp=sharing

Note: Before running, please change the model save folder to point to your directory with the model architecture config file and data folder to point to your data. Both of these are in the associated shell scripts. The code itself contains no absolute paths.

# Setup
To aid reproducibility and decrease setup time we provide a [Singularity Image](https://sylabs.io/singularity/) that contains all packages necessary to run the code without any further setup. The image is available on dropbox here: https://www.dropbox.com/s/ey74fiw4uquww0n/tfv1.13_tcmalloc.simg?dl=0


# Finetuning Workflow

## Static assets (never duplicated across experiments)
| Asset | Description |
|---|---|
| `models/net1/` | Original pretrained weights — read-only |
| `data/` | Original 48kHz TFRecords — read-only. Download from [Dropbox](https://www.dropbox.com/sh/af6vaotxt41i7pe/AACfTzMxMLfv-Edmn33S4gTpa?dl=0).|
| `echo_finetune/` | Finetuning data from [outputPB_ML_stims](https://www.dropbox.com/scl/fo/4r8744qlm1dqjx64nb1hb/AM0RiUErkDmXxuIj80G6-d8?rlkey=cz8ovr4v85os7odmhxh27fls7&st=6lrussgp&dl=0). Create symbolic link to your dropbox directory "outputPB_ML_stims" such as: `ln -s /path/to/your/dropbox/outputPB_ML_stims echo_finetune` |
| `echo_finetune_tfrecords/` | Precedence-effect finetuning data — read-only. Create from `echo_finetune` using `create_echo_finetune_tfrecords.py` |
| `data/data_original_eval.tfrecords` | Use `create_original_eval_shard.py` to create your own |
| `finetune_custom.py` | All options via CLI args — never duplicated |

## Experiment convention
Each experiment lives in `experiments/expNN_short_description/`:
```
experiments/expNN_name/
├── run.sh        ← exact command (this IS the config, committed to git)
├── notes.md      ← results, insights (committed to git)
├── checkpoints/  ← model.ckpt-* + best_val.ckpt (gitignored)
└── logs/         ← TensorBoard event files (gitignored)
    ├── train/
    ├── val/
    └── original/
```

## Experiment log

This is notes from an old run. Probably does not reflect the current state of the code.

| Exp | Description | val acc_0click | Notes |
|---|---|---|---|
| [exp01](https://github.com/znasif/BinauralLocalizationCNN/blob/08f4f832f81b7b77db52907f261e9f5c01c0bfc6/experiments/exp01_baseline_lr5e-5/notes.md) | Full finetune, lr=5e-5, 20ep | 19.57% | Baseline. Heavy overfitting. |

## One-time setup — create the eval shard from original data which we will use to track catastrophic forgetting (run once, in Docker)
```bash
# Inside Docker (after cd ../app and pip install):
python create_original_eval_shard.py --data_dir data --output data/data_original_eval.tfrecords --n_records 200
```

## Running an experiment (Docker)
```bash
docker run -it --gpus all -p 6006:6006 -v /mnt/d/Projects/BinauralLocalizationCNN:/app nvcr.io/nvidia/tensorflow:20.12-tf1-py3 bash
cd ../app
pip install --upgrade pip && pip install "numpy<1.20" "matplotlib<3.6" scipy seaborn
bash experiments/expNN_name/run.sh
```

## Monitoring (TensorBoard)
```bash
# Single experiment:
tensorboard --logdir /mnt/d/Projects/BinauralLocalizationCNN/experiments/expNN_name/logs/
# All experiments side-by-side:
tensorboard --logdir /mnt/d/Projects/BinauralLocalizationCNN/experiments/
# Open http://localhost:6006
```

## Available finetune_custom.py flags
```
--freeze_conv          Freeze all conv/BN layers; train only FC layers (reduces overfitting)
--weight_decay FLOAT   L2 weight decay (e.g. 1e-4)
--save_best_val        Also save best_val.ckpt when val acc_0click improves
--lr FLOAT             Learning rate (default 5e-5)
--epochs INT           Number of epochs (default 20)
--batch_size INT       Batch size (default 16)
--val_fraction FLOAT   Val split fraction (default 0.2)
```

# Inference only (no Docker needed)
```bash
conda activate rapids
python test_inference_tflite.py --model_file ./models/net1/model.tflite --wav_file echo/az-100_tdist100_Idist100_fs96000.wav
python test_inference_tflite.py --model_file ./experiments/exp02_alllayers_lr5e-5_frozenbn/model.tflite --wav_file echo_finetune/EE1/az5_ML_clickEcho_samples/wavs/az5_el20_1click_distRange_80-120cm_onsetRange_50-200ms_fs96000_samp100.wav
```

# Original Docker inference commands
```bash
docker run -it --gpus all -p 6006:6006 -v /mnt/d/Projects/BinauralLocalizationCNN:/app nvcr.io/nvidia/tensorflow:20.12-tf1-py3 bash
cd ../app
pip install --upgrade pip && pip install "numpy<1.20" "matplotlib<3.6" scipy seaborn
python test_inference_minimal.py --model_dir models/net1 --tfrecord data/train0.tfrecords --plot_output=./plots/ --use_gpu
python test_inference_minimal.py   --model_dir models/net1   --checkpoint experiments/exp02_alllayers_lr5e-5_frozenbn/checkpoints/model.ckpt-6   --tfrecord data/data_original_eval.tfrecords   --plot_output ./plots/   --use_gpu
python test_inference_minimal.py --model_dir models/net1 --wav_file=echo/az-100_tdist100_Idist100_fs96000.wav --plot_output=./plots/ --use_gpu
python test_inference_minimal.py --model_dir models/net1 --wav_folder=echo/ --plot_output=./plots/ --use_gpu
```

# Activation map visualisation (intermediate layer outputs)

`--activation_output` runs on the **first input sample only** and saves one PNG per
captured layer. Use it alongside any input mode (wav file, tfrecord, dummy).

```bash
# From a wav file (CPU, no Docker needed):
conda activate rapids
python test_inference_minimal.py \
    --model_dir models/net1 \
    --wav_file echo/az-100_tdist100_Idist100_fs96000.wav \
    --activation_output ./activation_maps/

# From a tfrecord sample (Docker, GPU):
python test_inference_minimal.py \
    --model_dir models/net1 \
    --tfrecord data/train0.tfrecords \
    --sample_index 0 \
    --activation_output ./activation_maps/ \
    --use_gpu

# With a finetuned checkpoint:
python test_inference_minimal.py \
    --model_dir models/net1 \
    --checkpoint experiments/exp02_alllayers_lr5e-5_frozenbn/checkpoints/model.ckpt-6 \
    --wav_file echo/az-100_tdist100_Idist100_fs96000.wav \
    --activation_output ./activation_maps/
```

## Output files

All PNGs are written to the directory given by `--activation_output`.
The filename pattern is `activation_<layer_name>.png` where `<layer_name>`
is the TF op name with `/` and `:` replaced by `_`.

> `--activation_output` only runs on the first sample even when multiple
> inputs are provided (e.g. `--wav_folder` or `--tfrecords_dir`). Prediction and
> aggregate plots for all other samples are unaffected.

Captured layer types and what they show:

| Layer type | What the image shows |
|---|---|
| `Relu` | Post-activation feature maps — grid of freq × time images, one per channel (up to 16 shown; channels sampled evenly if more exist) |
| `MaxPool` | Pooling outputs — same grid format but smaller spatial dimensions |
| `Reshape` | Flattened vector just before FC layers — bar chart of all 56 320 unit values + activation distribution histogram |
| `Softmax` | Final output — bar chart of all 504 class probabilities |

## Analysis output directories

| | `activation_maps/` | `tuning_analysis/` | `attribution_analysis/` |
|---|---|---|---|
| **Question** | What does each layer's output look like? | Which units are tuned to which directions? | Which input regions caused this prediction? |
| **Method** | Forward pass — raw activations | Forward pass × many samples — class-conditional mean activations | Backward pass — gradients from output to input |
| **Direction** | Input → forward → visualize layers | Input → forward → aggregate stats per unit | Output → backward → trace to input |
| **Granularity** | Single sample, all layers | All samples, per-unit aggregate | Per-sample + aggregate |
| **Content** | 1 PNG per layer (channel mosaic) | Tuning maps (7×72 per unit), selectivity indices, top-k mosaics, HTML report | Saliency overlays, Grad-CAM ladders, FC decomposition, attention profiles |
| **Script** | `test_inference_minimal.py` | `analyze_unit_tuning.py` | `analyze_attribution.py` |
| **Key data** | PNGs only | `tuning_maps.npz` (unit×az×el) | `attribution_data.npz` (mean saliency by sector) |
| **Answers** | "Layer 3 has 128 channels, here's what they look like" | "Unit 42 in pool\_2 fires most for azimuth=90°, elevation=10°" | "For this 90° sample, pool\_2 attended to freq bins 15–22 because of ILD cues" |

> **Note:** 
> Refer to [deepSTRF](https://github.com/urancon/deepSTRF/tree/develop) for pytorch and [Earshot](https://pmc.ncbi.nlm.nih.gov/articles/PMC12331064/#notes1) for Keras based implementations that are more recent and show a wide range of Stateless and Stateful architectures but not for Binaural Localization.

> **Note:** 
> Old localization based RNNs may be found by searching for DCASE Challenges such as [SELD-DCASE 2019](https://github.com/yinkalario/Two-Stage-Polyphonic-Sound-Event-Detection-and-Localization), [SELD-DCASE 2020](https://github.com/thomeou/General-network-architecture-for-sound-event-localization-and-detection) or [SELD-DCASE 2023](https://github.com/sharathadavanne/seld-dcase2023?tab=readme-ov-file).