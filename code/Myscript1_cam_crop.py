# -*- coding:UTF-8 -*-

import cv2
import numpy as np

def crop(url):
    img = cv2.imread(url)
    x = float(img.shape[0])
    y = float(img.shape[1])


    print x,y

    if x < 224 or y < 224:
        scale1 = x / y
        scale2 = y / x
        if scale1 < scale2:
            img = cv2.resize(img, (int(scale2 * 224),224))
        else:
            img = cv2.resize(img, (224, int(scale1 * 224)))
    x = img.shape[0]
    y = img.shape[1]


    print x,y

    step_x = (x - 224) / 35 + 1
    step_y = (y - 224) / 35 + 1

    if x > 451 and y > 451:
        step_x = (x - 224) / 70 + 1
        step_y = (y - 224) / 70 + 1

    print step_x, step_y

    for i in range(step_x):
        for j in range(step_y):
            x = i * 35
            y = j * 35
            crop = img[x:x + 224, y:y + 224, :]
            crop = np.resize(crop,(1,224,224,3))

url1 = '/media/wangchen/newdata1/wangchen/dataSet/Indoor67/CAM/closet/8c_CAM.jpg'
url2 = '/media/wangchen/newdata1/wangchen/dataSet/Indoor67/CAM/closet/closet_design_lg_gif_CAM.jpg'
url3 = '/media/wangchen/newdata1/wangchen/dataSet/Indoor67/CAM/grocerystore/070707_15291_CAM.jpg'
crop(url3)
print 'Myscript 1 finish!'