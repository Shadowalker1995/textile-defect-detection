#!/bin/bash

python confusion_matrix.py -m "CNN2" -bs 128
python confusion_matrix.py -m "GoogleNet" -bs 16
python confusion_matrix.py -m "DenseNet" -bs 16
python confusion_matrix.py -m "ResNet" -bs 32
python confusion_matrix.py -m "ResNetPreAct" -bs 32