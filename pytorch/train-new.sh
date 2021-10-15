#!/bin/bash

# echo ===========
# echo "CNN2"
# echo ===========
# python train.py -m "CNN2" -bs 128 -e 150 -lr 1e-3 -wd 1E-4 -opt "Adam" -rs 42 -w 1
# python train.py -m "CNN2" -bs 128 -e 150 -lr 1e-3 -wd 1E-4 -opt "Adam" -rs 2021 -w 1
# python train.py -m "CNN2" -bs 128 -e 150 -lr 1e-3 -wd 1E-4 -opt "Adam" -rs 369 -w 1
# python train.py -m "CNN2" -bs 128 -e 150 -lr 1e-3 -wd 1E-4 -opt "Adam" -rs 218 -w 1
# python train.py -m "CNN2" -bs 128 -e 150 -lr 1e-3 -wd 1E-4 -opt "Adam" -rs 1995 -w 1

#
# echo ===========
# echo "GoogleNet"
# echo ===========
# python train.py -m "GoogleNet" -bs 16 -e 150 -lr 1E-3 -wd 1E-4 -opt "Adam" -rs 42 -w 32
# python train.py -m "GoogleNet" -bs 16 -e 150 -lr 1E-3 -wd 1E-4 -opt "Adam" -rs 2021 -w 32
# python train.py -m "GoogleNet" -bs 16 -e 150 -lr 1E-3 -wd 1E-4 -opt "Adam" -rs 369 -w 32
# python train.py -m "GoogleNet" -bs 16 -e 150 -lr 1E-3 -wd 1E-4 -opt "Adam" -rs 218 -w 32
# python train.py -m "GoogleNet" -bs 16 -e 150 -lr 1E-3 -wd 1E-4 -opt "Adam" -rs 1995 -w 32

echo ===========
echo "ResNet"
echo ===========
python train.py -m "ResNet" -bs 32 -e 150 -lr 0.1 -wd 1E-4 -mo 0.9 -opt "SGD" -rs 42 -w 32
# python train.py -m "ResNet" -bs 32 -e 150 -lr 0.1 -wd 1E-4 -mo 0.9 -opt "SGD" -rs 2021 -w 32
# python train.py -m "ResNet" -bs 32 -e 150 -lr 0.1 -wd 1E-4 -mo 0.9 -opt "SGD" -rs 369 -w 32
# python train.py -m "ResNet" -bs 32 -e 150 -lr 0.1 -wd 1E-4 -mo 0.9 -opt "SGD" -rs 218 -w 32
# python train.py -m "ResNet" -bs 32 -e 150 -lr 0.1 -wd 1E-4 -mo 0.9 -opt "SGD" -rs 1995 -w 32

echo ===========
echo "ResNetPreAct"
echo ===========
python train.py -m "ResNetPreAct" -bs 32 -e 150 -lr 0.1 -wd 1E-4 -mo 0.9 -opt "SGD" -rs 42 -w 32
# python train.py -m "ResNetPreAct" -bs 32 -e 150 -lr 0.1 -wd 1E-4 -mo 0.9 -opt "SGD" -rs 2021 -w 32
# python train.py -m "ResNetPreAct" -bs 32 -e 150 -lr 0.1 -wd 1E-4 -mo 0.9 -opt "SGD" -rs 369 -w 32
# python train.py -m "ResNetPreAct" -bs 32 -e 150 -lr 0.1 -wd 1E-4 -mo 0.9 -opt "SGD" -rs 218 -w 32
# python train.py -m "ResNetPreAct" -bs 32 -e 150 -lr 0.1 -wd 1E-4 -mo 0.9 -opt "SGD" -rs 1995 -w 32

# echo ===========
# echo "DenseNet"
# echo ===========
# python train.py -m "DenseNet" -bs 16 -e 200 -lr 1E-3 -wd 1E-4 -opt "Adam" -rs 42 -w 32
# python train.py -m "DenseNet" -bs 16 -e 200 -lr 1E-3 -wd 1E-4 -opt "Adam" -rs 2021 -w 32
# python train.py -m "DenseNet" -bs 16 -e 200 -lr 1E-3 -wd 1E-4 -opt "Adam" -rs 369 -w 32
# python train.py -m "DenseNet" -bs 16 -e 200 -lr 1E-3 -wd 1E-4 -opt "Adam" -rs 218 -w 32
# python train.py -m "DenseNet" -bs 16 -e 200 -lr 1E-3 -wd 1E-4 -opt "Adam" -rs 1995 -w 32
