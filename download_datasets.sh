#!/bin/bash


if [ ! -d "datasets/MARS/raw_images_front" ]; then

    FRONT_FILE_LINK="https://data.caltech.edu/records/j1ww1-mdc55/files/raw_images_front.zip?download=1"
    FRONT_FILE_NAME="raw_images_front.zip"

    echo "Downloading: raw_images_front.zip"
    wget ${FRONT_FILE_LINK} -O datasets/MARS/${FRONT_FILE_NAME}

    echo "Unzipping: raw_images_front.zip"
    unzip datasets/MARS/${FRONT_FILE_NAME} -d datasets/MARS
    rm datasets/MARS/${FRONT_FILE_NAME}

fi


if [ ! -d "datasets/MARS/raw_images_top" ]; then

    TOP_FILE_LINK="https://data.caltech.edu/records/j1ww1-mdc55/files/raw_images_top.zip?download=1"
    TOP_FILE_NAME="raw_images_top.zip"

    echo "Downloading: raw_images_top.zip"
    wget ${TOP_FILE_LINK} -O datasets/MARS/${TOP_FILE_NAME}

    echo "Unzipping: raw_images_top.zip"
    unzip datasets/MARS/${TOP_FILE_NAME} -d datasets/MARS

    rm datasets/MARS/${TOP_FILE_NAME}
fi


echo "Done."
