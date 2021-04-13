#!/bin/bash

echo ===========
echo "CNN"
echo ===========
#python train.py -m "CNN" -bs 128 -e 300 -lr 8E-4 -wd 0 -opt "Adam" -rs 42 -w 0

echo ===========
echo "Inception_pretrain_full"
echo ===========
#python train.py -m "Inception" -sn "Inception_pretrain_full" -bs 64 -e 300 -lr 8E-4 -wd 0 -opt "Adam" --pretrained --full-train -rs 42 -w 0
echo ===========
echo "Inception_no-pretrain_full"
echo ===========
python train.py -m "Inception" -sn "Inception_no-pretrain_full" -bs 64 -e 300 -lr 8E-4 -wd 0 -opt "Adam" --no-pretrained --full-train -rs 42 -w 0
echo ===========
echo "Inception_pretrain_no-full"
echo ===========
python train.py -m "Inception" -sn "Inception_pretrain_no-full" -bs 64 -e 300 -lr 8E-4 -wd 0 -opt "Adam" --pretrained --no-full-train -rs 42 -w 0
echo ===========
echo "Inception_no-pretrain_no-full"
echo ===========
python train.py -m "Inception" -sn "Inception_no-pretrain_no-full" -bs 64 -e 300 -lr 8E-4 -wd 0 -opt "Adam" --no-pretrained --no-full-train -rs 42 -w 0

echo ===========
echo "GoogleNet"
echo ===========
python train.py -m "GoogleNet" -bs 16 -e 300 -lr 1E-3 -wd 1E-4 -opt "Adam" -rs 42 -w 0
echo ===========
echo "ResNet"
echo ===========
python train.py -m "ResNet" -bs 32 -e 300 -lr 0.1 -wd 1E-4 -mo 0.9 -opt "SGD" -rs 42 -w 0
echo ===========
echo "ResNetPreAct"
echo ===========
python train.py -m "ResNetPreAct" -bs 32 -e 300 -lr 0.1 -wd 1E-4 -mo 0.9 -opt "SGD" -rs 42 -w 0
echo ===========
echo "DenseNet"
echo ===========
python train.py -m "DenseNet" -bs 16 -e 300 -lr 1E-3 -wd 1E-4 -opt "Adam" -rs 42 -w 0

