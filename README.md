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
| `data/` | Original 48kHz TFRecords — read-only |
| `echo_finetune_tfrecords/` | Precedence-effect finetuning data — read-only |
| `data/data_original_eval.tfrecords` | Pre-downsampled eval shard (created once below) |
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

| Exp | Description | val acc_0click | Notes |
|---|---|---|---|
| [exp01](experiments/exp01_baseline_lr5e-5/notes.md) | Full finetune, lr=5e-5, 20ep | 19.57% | Baseline. Heavy overfitting. |

## One-time setup — create the forgetting eval shard (run once, in Docker)
```bash
# Inside Docker (after cd ../app and pip install):
python create_original_eval_shard.py --data_dir data --output data/data_original_eval.tfrecords --n_records 200
```

## Running an experiment (Docker)
```bash
docker run -it --gpus all -v /mnt/d/Projects/BinauralLocalizationCNN:/app nvcr.io/nvidia/tensorflow:20.12-tf1-py3 bash
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
```

# Original Docker inference commands
```bash
docker run -it --gpus all -p 6006:6006 -v /mnt/d/Projects/BinauralLocalizationCNN:/app nvcr.io/nvidia/tensorflow:20.12-tf1-py3 bash
cd ../app
pip install --upgrade pip && pip install "numpy<1.20" "matplotlib<3.6" scipy seaborn
python test_inference_minimal.py --model_dir models/net1 --tfrecord data/train0.tfrecords --plot_output=./plots/ --use_gpu
python test_inference_minimal.py   --model_dir models/net1   --checkpoint experiments/exp01_baseline_lr5e-5/checkpoints/model.ckpt-20   --tfrecord data/data_original_eval.tfrecords   --plot_output ./plots/   --use_gpu
python test_inference_minimal.py --model_dir models/net1 --wav_file=echo/az-100_tdist100_Idist100_fs96000.wav --plot_output=./plots/ --use_gpu
python test_inference_minimal.py --model_dir models/net1 --wav_folder=echo/ --plot_output=./plots/ --use_gpu
```