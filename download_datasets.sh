#!/bin/bash

if [ ! -d "datasets/MARS/raw_images_front" ]; then

    RAW_IMAGE_FRONT_ZIPFILE_LINK="https://data.caltech.edu/records/j1ww1-mdc55/files/raw_images_front.zip?download=1"
    echo "Downloading: raw_images_front.zip"
    wget ${RAW_IMAGE_FRONT_ZIPFILE_LINK} -O datasets/MARS/raw_images_front.zip
    echo "Unzipping: raw_images_front.zip"
    unzip datasets/MARS/raw_images_front.zip -d datasets/MARS
    rm datasets/MARS/raw_images_front.zip

    KEYPOINTS_FRONT_JSONFILE_LINK="https://data.caltech.edu/records/j1ww1-mdc55/files/MARS_keypoints_front.json?download=1"
    echo "Downloading: MARS_keypoints_front.json"
    wget ${KEYPOINTS_FRONT_JSONFILE_LINK} -O datasets/MARS

fi

if [ ! -d "datasets/MARS/raw_images_top" ]; then

    RAW_IMAGE_TOP_ZIPFILE_LINK="https://data.caltech.edu/records/j1ww1-mdc55/files/raw_images_top.zip?download=1"
    echo "Downloading: raw_images_top.zip"
    wget ${RAW_IMAGE_TOP_ZIPFILE_LINK} -O datasets/MARS/raw_images_top.zip
    echo "Unzipping: raw_images_top.zip"
    unzip datasets/MARS/raw_images_top.zip -d datasets/MARS
    rm datasets/MARS/raw_images_top.zip

    KEYPOINTS_TOP_JSONFILE_LINK="https://data.caltech.edu/records/j1ww1-mdc55/files/MARS_keypoints_top.json?download=1"
    echo "Downloading: MARS_keypoints_front.json"
    wget ${KEYPOINTS_TOP_JSONFILE_LINK} -O datasets/MARS

fi

echo "Done."
