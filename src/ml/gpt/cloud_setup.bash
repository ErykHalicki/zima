#!/bin/bash
set -e

apt-get update && apt-get install -y git unzip

if [ ! -f /usr/local/bin/aws ]; then
    echo "Installing AWS CLI..."
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    ./aws/install
    rm -rf aws awscliv2.zip
else
    echo "AWS CLI already installed, skipping installation"
fi

aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID
aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
aws configure set default.region us-east-2

if [ ! -d "zima" ]; then
    echo "Cloning zima repository..."
    git clone https://github.com/ErykHalicki/zima.git
else
    echo "Zima repository already exists, skipping clone"
fi
cd zima/src/ml/gpt

mkdir -p ~/datasets
mkdir -p ~/model_weights

if [ ! -f ~/datasets/wikipedia_Machine_learning.hdf5 ]; then
    echo "Downloading dataset..."
    aws s3 cp s3://zima-data/datasets/wikipedia_Machine_learning.hdf5 ~/datasets/wikipedia_Machine_learning.hdf5
else
    echo "Dataset already exists, skipping download"
fi

pip install -r requirements.txt

python train.py --config configs/train_config.yaml

