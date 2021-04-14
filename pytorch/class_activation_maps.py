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
import sys
import pickle
import argparse
from PIL import Image

from Model import Trainer
from utils import get_cam


parser = argparse.ArgumentParser()

# Model
parser.add_argument("-m", dest="model", type=str, default="CNN2",
                    help="Model Name, e.g. CNN|Inception|GoogleNet|ResNet|ResNetPreAct|DenseNet (Default: CNN2)")
parser.add_argument("-sn", dest="save_name", type=str, default="",
                    help="Specify the file name for saving model! (Default: "", i.e. Disabled)")

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
        print(f"Can not found pretrained model at {pretrained_filename}, exit...")
        sys.exit()

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
    SAVE_NAME = args.save_name if args.save_name != "" else MODEL_NAME
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

    SAVE_NAME = 'DenseNet'

    if not os.path.exists(f'results/{SAVE_NAME}'):
        os.mkdir(f'results/{SAVE_NAME}')

    model = load_model(SAVE_NAME).cuda()
    model = model.model

    # model = model.cpu()
    # parm = {}
    # for name, parameters in model.named_parameters():
    #     print(name, ':', parameters.size())
    #     parm[name] = parameters.detach().numpy()

    final_conv = ''
    features_blobs = []
    if SAVE_NAME == 'CNN2':
        # hook the feature extractor
        model.conv_5.register_forward_hook(hook_feature)
    elif SAVE_NAME in ['Inception_no-pretrain_full', 'Inception_pretrain_full', 'Inception_pretrain_no-full']:
        model.main_bone.Mixed_7c.register_forward_hook(hook_feature)
    elif SAVE_NAME == 'GoogleNet':
        model.inception_blocks._modules.get('9').register_forward_hook(hook_feature)
    elif SAVE_NAME in ['ResNet', 'ResNetPreAct']:
        model.blocks._modules.get('8').register_forward_hook(hook_feature)
    elif SAVE_NAME == 'DenseNet':
        model.blocks._modules.get('6').register_forward_hook(hook_feature)

    # random select
    # test_data = ImageFolder(root=TEST_DATA_PATH, transform=transform_val)
    # random_index = np.random.randint(len(test_data))
    # img_path = test_data.samples[random_index][0]
    # classes = test_data.classes
    #
    # target = test_data[random_index][1]
    # classname = test_data.classes[target]
    # print('the ground truth class is', classname)
    #
    # img = test_data[random_index][0]
    # get_cam(model, features_blobs, img, classes, img_path, SAVE_NAME)

    # loop through
    classes = ['Defect-Free',
               'DOG',
               'Broken-Pick',
               'Creases',
               'Kinky-Filling',
               'Transfer-Knot',
               'Stand-Indicator',
               'End-Out']
    for img_name in os.listdir('./samples'):
        classname = img_name
        print('the ground truth class is', classname)
        img_path = os.path.join('./samples', img_name)
        img = Image.open(img_path)
        img = transform_val(img)
        get_cam(model, features_blobs, img, classes, img_path, SAVE_NAME)

