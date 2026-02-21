# BinauralLocalizationCNN
**Code to create networks that localize sounds sources in 3D environments**


1. Main training/testing python script is `call_model_training_valid_pad_francl.py`. 
    * This script is responsible for processing the experiment parameters, validating the model folder, saving a copy of the experiment parameters there and ensuring the same folder isn't used for two different training rounds.
    * An example set of parameeters can be found for testing in `slurm_CNN_spherical_testing_francl.sh` and in `slurm_CNN_spherical_training_francl.sh`.
3. Networks weights can be downloaded at: https://www.dropbox.com/sh/af6vaotxt41i7pe/AACfTzMxMLfv-Edmn33S4gTpa?dl=0

2. The model input should be nervegrams with their associated metadata saved into tensorflow records. The cochlear model we use is the [PyCochleagram package ](https://github.com/mcdermottLab/pycochleagram). We have a wrapper to transform stereo `.wav` files into the proper input available here: https://github.com/afrancl/BinauralDataGen
    * The precomputed training data that were used to train the original model can be downloaded here: https://drive.google.com/drive/folders/1pUyMmkurxEWFeofbLKy5-QVWrcMbM8Wz?usp=sharing

Note: Before running, please change the model save folder to point to your directory with the model architecture config file and data folder to point to your data. Both of these are in the associated shell scripts. The code itself contains no absolute paths.

# Setup
To aid reproducibility and decrease setup time we provide a [Singularity Image](https://sylabs.io/singularity/) that contains all packages necessary to run the code without any further setup. The image is available on dropbox here: https://www.dropbox.com/s/ey74fiw4uquww0n/tfv1.13_tcmalloc.simg?dl=0


# Finetuning Workflow

## One-time setup — create the forgetting eval shard (run once, in Docker)
```bash
# Inside Docker (after cd ../app and pip install):
python create_original_eval_shard.py --data_dir data --output data/data_original_eval.tfrecords --n_records 200
```
This downsamples 200 records from the original 48kHz data to 8kHz and saves them as a TFRecord.
Only needed once; the file is reused on every subsequent run.

## Terminal 1 — Docker (GPU finetuning + forgetting eval, all-in-one)
```bash
docker run -it --gpus all -v /mnt/d/Projects/BinauralLocalizationCNN:/app nvcr.io/nvidia/tensorflow:20.12-tf1-py3 bash
cd ../app
pip install --upgrade pip && pip install "numpy<1.20" "matplotlib<3.6" scipy seaborn
# Finetune + forgetting eval each epoch (acc_original written to logs/original/)
python finetune_custom.py --tfrecords_dir ./echo_finetune_tfrecords/ --model_dir models/net1 --output_dir checkpoints_finetuned --log_dir logs --epochs 20 --batch_size 16 --lr 5e-5
```

## Terminal 2 — TensorBoard
```bash
tensorboard --logdir /mnt/d/Projects/BinauralLocalizationCNN/logs/
# Open http://localhost:6006 — shows train, val, and original (acc_original) all epoch-aligned
```

# Inference only (no Docker needed)
```bash
conda activate rapids
python test_inference_tflite.py --model_file ./models/net1/model.tflite --wav_file echo/az-100_tdist100_Idist100_fs96000.wav
```

# Original Docker inference commands
```bash
docker run -it --gpus all -v /mnt/d/Projects/BinauralLocalizationCNN:/app nvcr.io/nvidia/tensorflow:20.12-tf1-py3 bash
cd ../app
pip install --upgrade pip && pip install "numpy<1.20" "matplotlib<3.6" scipy seaborn
python test_inference_minimal.py --model_dir models/net1 --tfrecord data/train0.tfrecords --plot_output=./plots/ --use_gpu
python test_inference_minimal.py --model_dir models/net1 --wav_file=echo/az-100_tdist100_Idist100_fs96000.wav --plot_output=./plots/ --use_gpu
python test_inference_minimal.py --model_dir models/net1 --wav_folder=echo/ --plot_output=./plots/ --use_gpu
```