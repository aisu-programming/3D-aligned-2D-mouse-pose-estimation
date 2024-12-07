# Aligning 3D Latent Space for better 2D Mouse Pose Estimation

(Updated on 2024/12/06)

Utilized InfoNCE loss to align latent spaces from different-view 2D image inputs, allowing the model to learn a sense of 3D from 2D images.

## Details

Datasets are stored in _datasets_.

Class objects for loading/augmenting images in datasets are stored in _data_.

Models such as Scalable UNet are stored in _models_.

Use _download\_datasets.sh_ to download images of MARS dataset.

Use _build\_resized\_PE\_dataset\_from\_local\_YOLO.py_ to generate the cropped dataset for Top-down methods.

Use the scripts in _scripts_ or _train_ to train the pipeline (with hyperparameter search using wandb).

<!-- ## Memo

Now using only 500 images from MARS for training/validation, this can be adjust at _data/dataset.py:88_.

## (Deprecated) Memo for SLEAP

The directory started to train model has to be the same place used `convert_COCO_to_slp.py`. -->