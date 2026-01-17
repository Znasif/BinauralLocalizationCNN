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


# Docker instructions for Ampere (and Turing?) Architecture
```bash
docker run -it --gpus all -v /mnt/d/Projects/BinauralLocalizationCNN:/app   nvcr.io/nvidia/tensorflow:20.12-tf1-py3 bash
# Navigate to app
cd ../app
# After navigating to app:
pip install --upgrade pip && pip install "numpy<1.20" "matplotlib<3.6" scipy seaborn
# Then run the inference script. Sample index can be between 0 and 199:
python test_inference_minimal.py --model_dir models/net1 --tfrecord data/train0.tfrecords --sample_index=199 --use_gpu
```

# Lots of questions
1. What to do about the 48kHz and 96kHz sampling rate mismatch? Look at prepare_hrir.py for my current thinking.
2. Are HRIR supposed to sound like clicks?
3. For the hrir dataset Santani shared, like az0_tdist100_Idist100_fs96000.wav, azimuth is 0 but what is the elevation? Is it 0 for all? What is tdist and idist?
4. Should I use 'NaturalSound_X_realWorldAIREchoThief' in dropbox 'Training_dataset_for_ML_model' folder for dry sounds? Or https://github.com/karolpiczak/ESC-50 or does not matter? Or did Santani share the dry sounds?