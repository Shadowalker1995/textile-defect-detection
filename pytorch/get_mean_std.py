#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FileName:	get_mean_std.py
Author:	Zhu Zhan
Email:	henry664650770@gmail.com
Date:		2021-04-09 20:32:43
"""

import os
import numpy as np
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import pickle


class Dataloader:
    def __init__(self, root='./data/', resize=(256, 256), channel_num=3):
        self.dirs = ['train', 'val', 'test']

        self.means = [0, 0, 0]
        self.stdevs = [0, 0, 0]

        if channel_num == 3:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(resize),
                    transforms.CenterCrop(resize),
                    # from PIL to tensor and the range from [0,255] to [0,1]
                    transforms.ToTensor(),
                ]
            )
        elif channel_num == 1:
            self.transform = transforms.Compose(
                [
                    transforms.Grayscale(num_output_channels=3),
                    transforms.Resize(resize),
                    transforms.CenterCrop(resize),
                    # from PIL to tensor and the range from [0,255] to [0,1]
                    transforms.ToTensor(),
                ]
            )

        self.dataset = {x: ImageFolder(os.path.join(root, x), self.transform) for x in self.dirs}

    def get_mean_std(self, type, mean_std_path):
        """
        calculate the mean and std of the dataset
        :param type: datatype, such as 'train', 'test', 'testing'
        :param mean_std_path: saved path
        :return:
        """
        num_imgs = len(self.dataset[type])
        for data in self.dataset[type]:
            img = data[0]
            for i in range(3):
                self.means[i] += img[i, :, :].mean()
                self.stdevs[i] += img[i, :, :].std()

        self.means = np.asarray(self.means) / num_imgs
        self.stdevs = np.asarray(self.stdevs) / num_imgs

        print("{} : normMean = {}".format(type, self.means))
        print("{} : normstdevs = {}".format(type, self.stdevs))

        with open(mean_std_path, 'wb') as f:
            pickle.dump(self.means, f)
            pickle.dump(self.stdevs, f)
            print('pickle done')


if __name__ == "__main__":
    root_path = "../data/8Classes-9041_hist/"
    dataloader = Dataloader(root=root_path, resize=(200, 200), channel_num=3)
    for x in dataloader.dirs:
        mean_std_path = root_path + 'mean_std_value_' + x + '.pkl'
        dataloader.get_mean_std(x, mean_std_path)
