#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FileName:	utils.py
Author:	Zhu Zhan
Email:	henry664650770@gmail.com
Date:		2021-04-13 15:50:23
"""

from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2


# generate class activation mapping for the top1 prediction
def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (200, 200)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


def get_cam(model, features_blobs, img_tensor, classes, img_path, model_name):
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
    cv2.imwrite(f'results/{model_name}/{classes[idx[0].item()]}_CAM.jpg', result)
    cv2.imwrite(f'results/{model_name}/{classes[idx[0].item()]}.jpg', img)
