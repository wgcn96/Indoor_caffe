# -*- coding: UTF-8 -*-

import cv2
import numpy as np
import os

from math import *
from time import time

# the global variables
labeldata = ["airport_inside", "artstudio", "auditorium", "bakery", "bar",
             "bathroom", "bedroom", "bookstore", "bowling", "buffet",
             "casino", "children_room", "church_inside", "classroom", "cloister",
             "closet", "clothingstore", "computerroom", "concert_hall",
             "corridor", "deli", "dentaloffice", "dining_room", "elevator",
             "fastfood_restaurant", "florist", "gameroom", "garage", "greenhouse",
             "grocerystore", "gym", "hairsalon", "hospitalroom", "inside_bus",
             "inside_subway", "jewelleryshop", "kindergarden", "kitchen", "laboratorywet",
             "laundromat", "library", "livingroom", "lobby", "locker_room", "mall",
             "meeting_room", "movietheater", "museum", "nursery", "office", "operating_room",
             "pantry", "poolinside", "prisoncell", "restaurant", "restaurant_kitchen",
             "shoeshop", "stairscase", "studiomusic", "subway", "toystore",
             "trainstation", "tv_studio", "videostore", "waitingroom",
             "warehouse", "winecellar"]

file_root = '/home/wangchen/DataSet/Indoor67/Images/'
train_file_url = '/home/wangchen/DataSet/Indoor67/Images/TrainImages.txt'
test_file_url = '/home/wangchen/DataSet/Indoor67/Images/TestImages.txt'

distortedImageRoot = '/media/wangchen/newdata1/wangchen/dataSet/Indoor67/distorted_4/'


def getLabel(label_str):
    for i in range(len(labeldata)):
        if label_str == labeldata[i]:
            return i
    return -1


def loadFromFile(fileURL):
    urlList = []
    labelList = []
    file = open(fileURL, 'r')
    while True:
        line = file.readline()
        if line == '':
            break
        line = line.strip()
        pos = line.find('/')
        label_str = line[:pos]
        curLabel = getLabel(label_str)
        if curLabel == -1:
            raise ValueError('label or url error!')
        urlList.append(line)
        labelList.append(curLabel)
    return urlList, labelList


def _rotate(image, angle):
    (h, w) = image.shape[:2]
    anglePi = angle * pi / 180.0
    cosA = cos(anglePi)
    sinA = sin(anglePi)
    X1 = abs(0.5 * h * cosA + 0.5 * w * sinA)
    X2 = abs(0.5 * h * cosA - 0.5 * w * sinA)
    Y1 = abs(-0.5 * h * sinA + 0.5 * w * cosA)
    Y2 = abs(-0.5 * h * sinA - 0.5 * w * cosA)
    H = 2 * max(Y1, Y2)
    W = 2 * max(X1, X2)
    scale1 = H / h
    scale2 = W / w
    scale = max(scale1, scale2)
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def _flip(image):
    return cv2.flip(image, 1)


# ------------------------
# 通过No_root_url读入一张图片，distorted后 写图片并记录url label 的info
# 返回 info List
# ------------------------
def loadImageAndDistorted(currentImageURL_NoRoot, label, distortedFlag):
    infoList = []

    imageURL = file_root + currentImageURL_NoRoot

    pos = currentImageURL_NoRoot.find('/')
    imageClass = currentImageURL_NoRoot[:pos]
    imageName = currentImageURL_NoRoot[pos + 1:-4]
    imageFolder = imageClass + '/'

    ima = cv2.imread(imageURL)

    image = []
    newImageURL = []

    # resize 224
    currentIma = cv2.resize(ima, (224, 224))
    image.append(currentIma)
    currentImageURL = imageFolder + imageName + '_1.jpg'
    newImageURL.append(currentImageURL)

    # 水平镜像
    currentIma = _flip(ima)
    currentIma = cv2.resize(currentIma, (224, 224))
    image.append(currentIma)
    currentImageURL = imageFolder + imageName + '_2.jpg'
    newImageURL.append(currentImageURL)

    # 旋转30度
    currentIma = _rotate(ima, 30)
    currentIma = cv2.resize(currentIma, (224, 224))
    image.append(currentIma)
    currentImageURL = imageFolder + imageName + '_3.jpg'
    newImageURL.append(currentImageURL)

    # 旋转-30度
    currentIma = _rotate(ima, -30)
    currentIma = cv2.resize(currentIma, (224, 224))
    image.append(currentIma)
    currentImageURL = imageFolder + imageName + '_4.jpg'
    newImageURL.append(currentImageURL)

    if os.path.exists(distortedImageRoot + imageFolder) == False:
        os.mkdir(distortedImageRoot + imageFolder)

    if distortedFlag == 6:
        # 旋转45度
        currentIma = _rotate(ima, 45)
        currentIma = cv2.resize(currentIma, (224, 224))
        image.append(currentIma)
        currentImageURL = imageFolder + imageName + '_5.jpg'
        newImageURL.append(currentImageURL)

        # 旋转-45度
        currentIma = _rotate(ima, -45)
        currentIma = cv2.resize(currentIma, (224, 224))
        image.append(currentIma)
        currentImageURL = imageFolder + imageName + '_6.jpg'
        newImageURL.append(currentImageURL)

    for i in range(distortedFlag):
        cv2.imwrite(distortedImageRoot + newImageURL[i], image[i])
        # info = newImageURL[i] + ' ' + str(label) + '\n'
        info = newImageURL[i] + '\n'
        infoList.append(info)

    return infoList


def makeDataSet(traintestflag, distortedFlag):
    newtrainfile = 'TrainImages.txt'
    newtestfile = 'TestImages.txt'
    newtrainfile = distortedImageRoot + newtrainfile
    newtestfile = distortedImageRoot + newtestfile

    if traintestflag == 'train':
        loadurl = train_file_url
    else:
        loadurl = test_file_url

    urlList, labelList = loadFromFile(loadurl)

    infoList = []

    for i in range(len(urlList)):
        currentImageURL_NoRoot = urlList[i]
        currentLabel = labelList[i]
        tmpList = loadImageAndDistorted(currentImageURL_NoRoot, currentLabel, distortedFlag)
        infoList.extend(tmpList)

    if traintestflag == 'train':
        file = open(newtrainfile, 'w')
    else:
        file = open(newtestfile, 'w')

    file.writelines(infoList)
    file.close()


def checkResult(fileURl):
    count = 0
    file = open(fileURl, 'r')
    while True:
        line = file.readline()
        line = line.strip()
        if line == '':
            break
        count += 1
        currentURL = distortedImageRoot + line
        cv2.imread(currentURL)


if __name__ == '__main__':
    '''
    currentImageURL_NoRoot = 'gameroom/bt_132294gameroom2.jpg'
    label = 1
    distortedFlag = 4
    infoList = loadImageAndDistorted(currentImageURL_NoRoot,label,distortedFlag)
    print infoList
    '''
    # start = time()
    # makeDataSet(traintestflag='train',distortedFlag=4)
    # end = time()
    # seconds = end - start
    # print 'finish! total run: {} seconds.',seconds
    checkResult(distortedImageRoot + 'TrainImages.txt')
    print 'check finish!'
