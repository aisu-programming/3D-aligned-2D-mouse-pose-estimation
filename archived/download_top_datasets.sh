#!/bin/bash

TOP_FILE_LINK="https://data.caltech.edu/records/j1ww1-mdc55/files/raw_images_top.zip?download=1"
TOP_FILE_NAME="raw_images_top.zip"

if ! command -v gdown &> /dev/null
then
    echo "gdown not installed, installing gdown..."
    pip install gdown
fi

echo "Downloading..."
gdown ${TOP_FILE_LINK} -O ${TOP_FILE_NAME}

echo "Unzip..."
unzip ${TOP_FILE_NAME} -d datasets/MARS

rm ${TOP_FILE_NAME}

echo "Done."
