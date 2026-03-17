docker run -it --gpus all -p 6006:6006 \
  -v /mnt/d/Projects/BinauralLocalizationCNN:/app \
  -v "/mnt/d/ski.org Dropbox/Nasif Zaman/outputPB_ML_stims:/app/echo_finetune" \
  nvcr.io/nvidia/tensorflow:20.12-tf1-py3 bash