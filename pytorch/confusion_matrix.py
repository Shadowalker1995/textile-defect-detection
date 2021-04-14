#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FileName:	confusion_matrix.py
Author:	Zhu Zhan
Email:	henry664650770@gmail.com
Date:		2021-04-14 15:31:43
"""

import torch
from torch.nn import functional as F
import argparse
from utils import load_data, load_model, plot_confusion_matrix, get_confusion_matrix


parser = argparse.ArgumentParser()

# Model
parser.add_argument("-m", dest="model", type=str, default="CNN2",
                    help="Model Name, e.g. CNN|Inception|GoogleNet|ResNet|ResNetPreAct|DenseNet (Default: CNN2)")
parser.add_argument("-sn", dest="save_name", type=str, default="",
                    help="Specify the file name for saving model! (Default: "", i.e. Disabled)")

# Hyperparameters
parser.add_argument("-bs", dest="batch_size", type=int, default=32, help="Batch Size (Default: 32)")
parser.add_argument("-c", dest="num_classes", type=int, default=8, help="Number of Classes (Default: 8)")
parser.add_argument("-is", dest="img_size", type=int, default=200, help="Input image size (Default: 200)")

# Miscellaneous
parser.add_argument("-w", dest="num_workers", type=int, default=0, help="Number of Workers (Default: 0)")
parser.add_argument("-gpu", dest="gpu", type=int, default=0, help="Which GPU to use? (Default: 0)")

args = parser.parse_args()


if __name__ == "__main__":
    # Setting the seed
    # pl.seed_everything(42)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    CKPT_PATH = "./ckpt/"

    NUM_CLASSES = args.num_classes
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    # CNN|CNN2|Inception|GoogleNet|ResNet|ResNetPreAct|DenseNet
    MODEL_NAME = args.model
    SAVE_NAME = args.save_name if args.save_name != "" else MODEL_NAME
    RESIZE = (args.img_size, args.img_size)
    PRETRAINED = False

    train_loader, val_loader, test_loader, train_data, _, _ = load_data(BATCH_SIZE, RESIZE, NUM_WORKERS)

    classes = train_data.classes

    SAVE_NAME = 'ResNetPreAct'

    model = load_model(CKPT_PATH, SAVE_NAME).cuda()

    train_cm = get_confusion_matrix(model, train_loader, NUM_CLASSES, device)
    val_cm = get_confusion_matrix(model, val_loader, NUM_CLASSES, device)
    test_cm = get_confusion_matrix(model, test_loader, NUM_CLASSES, device)

    plot_confusion_matrix(train_cm, classes, normalize=True, title='Train confusion matrix',
                          isSave=True, save_path=f'./results/{SAVE_NAME}/')
    plot_confusion_matrix(val_cm, classes, normalize=True, title='Validation confusion matrix',
                          isSave=True, save_path=f'./results/{SAVE_NAME}/')
    plot_confusion_matrix(test_cm, classes, normalize=True, title='Test confusion matrix',
                          isSave=True, save_path=f'./results/{SAVE_NAME}/')

    # get the per-class accuracy
    # print(confusion_matrix.diag()/confusion_matrix.sum(1))

