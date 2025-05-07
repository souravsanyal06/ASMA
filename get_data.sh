#!/bin/bash

echo "[ASMA] Downloading external assets..."

# Create folders if they don't exist
mkdir -p dataset pretrained videos

# Download dataset
echo "Downloading dataset.zip..."
gdown --id 1GMgwVvNk5HmPSz_MLvAc9g2P6_qAKfTd --output dataset.zip
unzip -q dataset.zip -d dataset && rm dataset.zip

# Download pretrained models
echo "Downloading pretrained.zip..."
gdown --id 1gHSTfwxWUhXHuuLAlRK94-qcY9jB_P58 --output pretrained.zip
unzip -q pretrained.zip -d pretrained && rm pretrained.zip

# Download videos
echo "Downloading videos.zip..."
gdown --id 1VBoGVONh2tylEz2IwCaIYHMSyFjxDfDZ --output videos.zip
unzip -q videos.zip -d videos && rm videos.zip

echo "[ASMA] All assets downloaded and extracted successfully."

