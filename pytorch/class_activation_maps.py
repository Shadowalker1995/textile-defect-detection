#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FileName:	class_activation_maps.py
Author:	Zhu Zhan
Email:	henry664650770@gmail.com
Date:		2021-04-12 22:48:19
"""

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import numpy as np
import os
import pickle
import argparse

from Model import Trainer
from utils import get_cam


parser = argparse.ArgumentParser()

# Model
parser.add_argument("-m", dest="model", type=str, default="CNN2",
                    help="Model Name, e.g. CNN|Inception|GoogleNet|ResNet|ResNetPreAct|DenseNet (Default: CNN2)")

# Hyperparameters
parser.add_argument("-c", dest="num_classes", type=int, default=8, help="Number of Classes (Default: 8)")
parser.add_argument("-is", dest="img_size", type=int, default=200, help="Input image size (Default: 200)")

# Miscellaneous
parser.add_argument("-gpu", dest="gpu", type=int, default=0, help="Which GPU to use? (Default: 0)")

args = parser.parse_args()


def load_model(model_name, **kwargs):
    """
    Inputs:
        model_name - Name of the model you want to run. Is used to look up the class in "model_dict"
        save_name (optional) - If specified, this name will be used for creating the checkpoint and logging directory.
    """
    pretrained_filename = os.path.join(CHECKPOINT_PATH, model_name + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = Trainer.load_from_checkpoint(pretrained_filename)
    else:
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        return

    return model


def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())


if __name__ == "__main__":
    # Setting the seed
    # pl.seed_everything(42)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    TEST_DATA_PATH = "../data/8Classes-9041/test/"
    MEAN_STD_PATH = "../data/8Classes-9041/mean_std_value_train.pkl"
    CHECKPOINT_PATH = "./ckpt/"

    NUM_CLASSES = args.num_classes
    # CNN|CNN2|Inception|GoogleNet|ResNet|ResNetPreAct|DenseNet
    MODEL_NAME = args.model
    RESIZE = (args.img_size, args.img_size)
    PRETRAINED = False

    if os.path.exists(MEAN_STD_PATH):
        with open(MEAN_STD_PATH, 'rb') as f:
            MEAN = pickle.load(f)
            STD = pickle.load(f)
            print('MEAN and STD load done')

    transform_val = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(RESIZE),
                transforms.CenterCrop(RESIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]
        )

    test_data = ImageFolder(root=TEST_DATA_PATH, transform=transform_val)

    print("Number of test samples: ", len(test_data))
    # classes are detected by folder structure
    print("Detected Classes are: ", test_data.class_to_idx)

    model = load_model(MODEL_NAME).cuda()
    model = model._modules.get('model')
    final_conv = ''
    if MODEL_NAME == 'CNN2':
        final_conv = 'conv_5'

    # hook the feature extractor
    features_blobs = []
    model._modules.get(final_conv).register_forward_hook(hook_feature)

    random_index = np.random.randint(len(test_data))
    img_path = test_data.samples[random_index][0]
    classes = test_data.classes

    target = test_data[random_index][1]
    classname = test_data.classes[target]
    print('the ground truth class is', classname)

    img = test_data[random_index][0]

    get_cam(model, features_blobs, img, classes, img_path)

    # parm = {}
    # for name, parameters in model.named_parameters():
    #     print(name, ':', parameters.size())
    #     parm[name] = parameters.detach().numpy()

