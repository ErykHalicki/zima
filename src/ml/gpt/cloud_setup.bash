#!/bin/bash
set -e 

apt-get update && apt-get install -y git unzip

curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install

aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID
aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
aws configure set default.region us-east-2

git clone https://github.com/ErykHalicki/zima.git
cd zima/src/ml/gpt

mkdir ~/datasets
mkdir ~/model_weights
aws s3 cp s3://zima-data/datasets/wikipedia_Monkey.hdf5 ~/datasets/wikipedia_Monkey.hdf5

pip install -r requirements.txt

python train.py --config configs/train_config.yaml

