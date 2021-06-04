#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FileName:	utils.py
Author:	Zhu Zhan
Email:	henry664650770@gmail.com
Date:		2021-04-13 15:50:23
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import numpy as np
import os
import cv2
import sys
import itertools
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

from Model import Trainer
from few_shot.models import get_few_shot_encoder


def returnCAM(feature_conv, weight_softmax, class_idx):
    """generate class activation mapping for the top1 prediction"""
    # generate the class activation maps upsample to 256x256
    size_upsample = (200, 200)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


def get_cam(model, features_blobs, img_tensor, img_name, classes, img_path, model_name):
    model.eval()
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

    img_variable = Variable(img_tensor.unsqueeze(0)).cuda()
    logit = model(img_variable)

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)

    # output: the prediction
    for i in range(0, 8):
        line = '{:.3f} -> {}'.format(probs[i], classes[idx[i].item()])
        print(line)

    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0].item()])

    # render the CAM and output
    print(f'output {classes[idx[0].item()]}_CAM.jpg for the top1 prediction: {classes[idx[0].item()]}')
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    CAM = cv2.resize(CAMs[0], (width, height))
    heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    # cv2.imwrite(f'results/{model_name}/{classes[idx[0].item()]}_CAM.jpg', result)
    # cv2.imwrite(f'results/{model_name}/{classes[idx[0].item()]}.jpg', img)

    cv2.imwrite(f'results/{model_name}/{os.path.splitext(img_name)[0]}_CAM.jpg', result)
    cv2.imwrite(f'results/{model_name}/{os.path.splitext(img_name)[0]}.jpg', img)

    # cv2.imwrite(f'results/{model_name}/{classname}/{os.path.splitext(img_name)[0]}_CAM.jpg', result)
    # cv2.imwrite(f'results/{model_name}/{classname}/{os.path.splitext(img_name)[0]}.jpg', img)


def load_model(model_path, model_name, model_suffix='.ckpt', **kwargs):
    """
    Inputs:
        model_path - Path to the the directory that saving the models
        model_name - Name of the model you want to run. Is used to look up the class in "model_dict"
    """
    pretrained_filename = os.path.join(model_path, model_name + model_suffix)
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        if model_suffix == '.ckpt':
            model = Trainer.load_from_checkpoint(pretrained_filename)
        if model_suffix == '.pth':
            model = get_few_shot_encoder()
            # model = model[:-1]
            model = nn.Sequential(
                # model,
                *list(model.children())[:-2],
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, 8)
            )
            checkpoints = torch.load(pretrained_filename)
            model.load_state_dict(checkpoints)
            # model = nn.Sequential(*list(model.children())[:-2])
    else:
        print(f"Can not found pretrained model at {pretrained_filename}, exit...")
        sys.exit()

    return model


def get_confusion_matrix(model, data_loader, num_classes, device, verbose=False):
        print("Generating confusion matrix...")
        confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int32)
        with torch.no_grad():
            loop = tqdm(data_loader, total=len(data_loader), leave=True)
            for imgs, labels in loop:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=-1)
                for t, p in zip(labels.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
        if verbose:
            print(confusion_matrix)
        return confusion_matrix.numpy()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          isSave=False,
                          save_path='',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `mormaliza=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion metrix, without normalization")

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if isSave:
        plt.savefig(save_path + '{}.png'.format(title))
    plt.show()


def load_data(batch_size, resize, num_workers):
    TRAIN_DATA_PATH = "../data/8Classes-9041_hist/train/"
    VAL_DATA_PATH = "../data/8Classes-9041_hist/val/"
    TEST_DATA_PATH = "../data/8Classes-9041_hist/test/"
    MEAN_STD_PATH = "../data/8Classes-9041_hist/mean_std_value_train.pkl"

    if os.path.exists(MEAN_STD_PATH):
        with open(MEAN_STD_PATH, 'rb') as f:
            MEAN = pickle.load(f)
            STD = pickle.load(f)
            print('MEAN and STD load done')

    transform = transforms.Compose(
            [
                # transforms.Grayscale(num_output_channels=3),
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                # transforms.RandomRotation(0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]
        )

    transform_val = transforms.Compose(
            [
                # transforms.Grayscale(num_output_channels=3),
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]
        )
    train_data = ImageFolder(root=TRAIN_DATA_PATH, transform=transform)
    train_loader = data.DataLoader(dataset=train_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=num_workers,
                                   pin_memory=True)
    val_data = ImageFolder(root=VAL_DATA_PATH, transform=transform_val)
    val_loader = data.DataLoader(dataset=val_data,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 pin_memory=True)
    test_data = ImageFolder(root=TEST_DATA_PATH, transform=transform_val)
    test_loader = data.DataLoader(dataset=test_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True)

    return train_loader, val_loader, test_loader, train_data, val_data, test_data
