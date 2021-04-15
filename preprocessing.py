#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FileName:	preprocessing.py
Author:	Zhu Zhan
Email:	henry664650770@gmail.com
Date:		2021-04-15 14:21:19
"""

import os
import cv2
import numpy as np

path = 'data/8Classes-9041_hist/'

files = os.walk(path)
for path, _, filelist in files:
    # print(path, dir, filelist)
    for filename in filelist:
        if filename.endswith('.bmp'):
            filepath = os.path.join(path, filename)
            img = cv2.imread(filepath)
            b = img[:, :, 0]
            g = img[:, :, 1]
            r = img[:, :, 2]
            h_b = cv2.equalizeHist(b)
            h_g = cv2.equalizeHist(g)
            h_r = cv2.equalizeHist(r)
            dst_img = cv2.merge((h_b, h_g, h_r))
            # dst1 = np.hstack([b, g, r])
            # dst2 = np.hstack([h_b, h_g, h_r])
            # dst = np.vstack([dst1, dst2])
            # img = np.hstack([img, dst_img])
            cv2.imwrite(filepath, dst_img)
            print(filepath)
