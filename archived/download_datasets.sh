#!/bin/bash

FILE_ID="1IjPMSPC-T8bQd9_pXFG6N0Ig7sI2DOVC"
FILE_NAME="MARS_raw_images_front.zip"

if ! command -v gdown &> /dev/null
then
    echo "gdown not installed, installing gdown..."
    pip install gdown
fi

echo "Downloading..."
gdown https://drive.google.com/uc?id=${FILE_ID} -O ${FILE_NAME}

echo "Unzip..."
unzip ${FILE_NAME} -d datasets/MARS

rm ${FILE_NAME}

echo "Done."