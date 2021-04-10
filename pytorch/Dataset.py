#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FileName:	Dataset.py
Author:	Zhu Zhan
Email:	henry664650770@gmail.com
Date:		2021-04-09 19:52:48
"""

from torch.utils.data import Dataset
import os
from PIL import Image
import torch


class MyDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        img, target = self.images[index], self.labels[index]
        return img, target

    def __len__(self):
        return len(self.images)
