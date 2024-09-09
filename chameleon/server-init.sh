#!/usr/bin/env bash
set -e

# install dependencies
sudo apt update && sudo apt upgrade -y
sudo apt install --fix-missing python3 openssh-server openssh-client \
	git protobuf-compiler python3-pip wget ffmpeg -y

# clone tensorflow model garden
export REPO_DIR="$HOME"/TensorFlow/models
export RESEARCH_DIR="$REPO_DIR"/research
export APP_DIR="$HOME"/CE
python3 -m pip install --ignore-installed --upgrade tensorflow==2.5.0 pycocotools
mkdir "$HOME"/TensorFlow && cd "$_"
git clone git@github.com:tensorflow/models.git
cd models/research
protoc object_detection/protos/*.proto --python_out=.

# object_detection api
mkdir "$HOME"/cocoapi && cd "$_"
git clone git@github.com:cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools "$REPO_DIR"/research/
cd "$RESEARCH_DIR"
cp object_detection/packages/tf2/setup.py .
python3 -m pip install .

# get pretrained model
cd "$HOME"
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v2_512x512_coco17_tpu-8.tar.gz
tar -xvfz centernet_resnet50_v2_512x512_coco17_tpu-8.tar.gz
mv centernet_resnet50_v2_512x512_coco17_tpu-8 "$APP_DIR"

echo "export MODEL_PATH=$APP_DIR/centernet_resnet50_v2_512x512_coco17_tpu-8/saved_model" >> ~/.bashrc
source "$HOME"/.bashrc

sudo ufw allow 10000
