# CLIP-pose-estimation

(Updated on 2024/10/11)

Research for combining CLIP with classical pose estimation models

## Details

Datasets are stored in _datasets_.

Class objects for loading/augmenting images in datasets are stored in _data_.

Models such as Scalable UNet are stored in _models_.

Use _download\_datasets.sh_ to download images of MARS dataset.

Use _train.py_ to train a Scalable UNet, or use _train\_wandb.py_ to do auto hyperparameter search on Scalable UNet.

## Memo

Now using only 1500 images from MARS for training/validation, this can be adjust at _data/dataset.py:88_.

## (Deprecated) Memo for SLEAP

The directory started to train model has to be the same place used `convert_COCO_to_slp.py`.