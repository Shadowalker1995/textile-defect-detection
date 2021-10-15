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

import os
import numpy as np
import pickle
import argparse
from PIL import Image

from utils import get_cam, load_model


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


def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())


if __name__ == "__main__":
    # Setting the seed
    # pl.seed_everything(42)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    TEST_DATA_PATH = "../data/8Classes-9041_hist/test/"
    MEAN_STD_PATH = "../data/8Classes-9041_hist/mean_std_value_train.pkl"
    CKPT_PATH = "./ckpt/"

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

    transform_val_3c = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(RESIZE),
                transforms.CenterCrop(RESIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]
        )

    transform_val_1c = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(RESIZE),
                transforms.CenterCrop(RESIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN[0], std=STD[0]),
            ]
        )

    # SAVE_NAME = 'few-shot'
    # SAVE_NAME = 'CNN2'
    # SAVE_NAME = 'DenseNet'
    # SAVE_NAME = 'GoogleNet'
    # SAVE_NAME = 'ResNet'
    SAVE_NAME = 'ResNetPreAct'

    if not os.path.exists(f'results/{SAVE_NAME}'):
        os.mkdir(f'results/{SAVE_NAME}')

    if SAVE_NAME == 'few-shot':
        MODEL_NAME = 'Fabric_nt=4_kt=2_qt=12_nv=4_kv=2_qv=1_classifier'
        model = load_model(CKPT_PATH, MODEL_NAME, model_suffix='.pth').cuda()
    else:
        model = load_model(CKPT_PATH, SAVE_NAME).cuda()
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
    elif SAVE_NAME == 'few-shot':
        model._modules.get('4').register_forward_hook(hook_feature)

    # random select
    # test_data = ImageFolder(root=TEST_DATA_PATH, transform=transform_val)
    # random_index = np.random.randint(len(test_data))
    # img_path = test_data.samples[random_index][0]
    # print(img_path)
    # classes = test_data.classes
    #
    # target = test_data[random_index][1]
    # classname = test_data.classes[target]
    # print('the ground truth class is', classname)
    #
    # img = test_data[random_index][0]
    # get_cam(model, features_blobs, img, classes, img_path, SAVE_NAME)

    # # loop through
    # classes = ['Defect-Free',
    #            'DOG',
    #            'Broken-Pick',
    #            'Creases',
    #            'Kinky-Filling',
    #            'Transfer-Knot',
    #            'Stand-Indicator',
    #            'End-Out']
    # classname = 'Transfer-Knot'
    # for img_name in os.listdir(f'./samples/tmp/{classname}'):
    #     # classname = os.path.splitext(img_name)[0]
    #     # print('the ground truth class is', classname)
    #     img_path = os.path.join(f'./samples/tmp/{classname}', img_name)
    #     img = Image.open(img_path)
    #     if SAVE_NAME == 'few-shot':
    #         img = transform_val_1c(img)
    #     else:
    #         img = transform_val_3c(img)
    #     # get_cam(model, features_blobs, img, classes, img_path, SAVE_NAME)
    #     get_cam(model, features_blobs, img, img_name, classname, classes, img_path, SAVE_NAME)
    #     features_blobs = []

    # loop through
    classes = ['Defect-Free',
               'DOG',
               'Broken-Pick',
               'Creases',
               'Kinky-Filling',
               'Transfer-Knot',
               'Stand-Indicator',
               'End-Out']
    for img_name in os.listdir(f'./samples/histeq'):
        classname = os.path.splitext(img_name)[0]
        print('the ground truth class is', classname)
        img_path = os.path.join(f'./samples/histeq', img_name)
        img = Image.open(img_path)
        if SAVE_NAME == 'few-shot':
            img = transform_val_1c(img)
        else:
            img = transform_val_3c(img)
        # get_cam(model, features_blobs, img, classes, img_path, SAVE_NAME)
        get_cam(model, features_blobs, img, img_name, classes, img_path, SAVE_NAME)
        features_blobs = []
