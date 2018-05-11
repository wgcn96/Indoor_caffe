# -*- coding: UTF-8 -*-

import numpy as np
import cv2


def getCAMmap(activation, weights_LR):
    # print(activation.shape)

    if activation.shape[0] == 1:  # only one image
        n_feat, w, h = activation[0].shape
        act_vec = np.reshape(activation[0], [n_feat, w * h])
        n_top = weights_LR.shape[0]
        out = np.zeros([w, h, n_top])

        for t in range(n_top):
            weights_vec = np.reshape(weights_LR[t], [1, weights_LR[t].shape[0]])
            heatmap_vec = np.dot(weights_vec, act_vec)
            heatmap = np.reshape(np.squeeze(heatmap_vec), [w, h])
            out[:, :, t] = heatmap
    else:  # 10 images (over-sampling)
        raise Exception('Not implemented')

    return out


# 图片归一化
def im2double(im):
    return cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)


def py_map2jpg(imgmap, rang, colorMap):
    if rang is None:
        rang = [np.min(imgmap), np.max(imgmap)]

    heatmap_x = np.round(imgmap * 255).astype(np.uint8)

    return cv2.applyColorMap(heatmap_x, cv2.COLORMAP_JET)


# return heatmap_x

def get_threshold(img, thresholdRatio):
    return np.max(img) * thresholdRatio


def get_bbox(img, threshold):
    x1 = y1 = 0
    x2 = y2 = 0
    flag = False
    shape = img.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img[i][j] >= threshold:
                if not flag:
                    x1 = i
                    y1 = j
                    flag = True
                y1 = min(y1, j)
                x2 = max(x2, i)
                y2 = max(y2, j)
    return x1, y1, x2, y2


def auto_adap_bbox(img, thresholdRatio, coverAreaRatio, stepSize):
    times = 0
    while True:
        threshold = get_threshold(img, thresholdRatio)
        x1, y1, x2, y2 = get_bbox(img, threshold)
        flag1 = (x1, y1) == (0, 0) and (x2, y2) == (img.shape[0], img.shape[1])
        flag2 = float((x2 - x1) * (y2 - y1)) / float(img.shape[0] * img.shape[1])
        if flag1 or flag2 >= coverAreaRatio:
            thresholdRatio += stepSize
            times += 1
        else:
            break
        if times > 5:
            break
    return x1, y1, x2, y2
