# -*- coding: UTF-8 -*-


import cv2
import numpy as np

from sklearn import preprocessing

# the global variables
train_num = 5360
test_num = 1340
labnum = 67
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

file_root = '/media/wangchen/newdata1/wangchen/dataSet/Indoor67/Images/'
train_file_url = '/media/wangchen/newdata1/wangchen/dataSet/Indoor67/Images/TrainImages.txt'
test_file_url = '/media/wangchen/newdata1/wangchen/dataSet/Indoor67/Images/TestImages.txt'


# ------------------------------------------
# prepare data module start
def getLabel(label_str):
    for i in range(len(labeldata)):
        if label_str == labeldata[i]:
            return i
    return -1


def loadImage(imageURL):
    ima = cv2.imread(imageURL)
    ima = cv2.resize(ima, (224, 224))  # 图像像素调整 ——》224*224
    ima = np.asarray(ima, dtype='float32') / 255.
    # cv2.cvtColor()
    ima = ima.transpose(2, 0, 1)  # 这张图片的格式为(h,w,rgb), 然后想办法交换成(rgb,h,w)
    return ima


def loadImageFromFile(fileURL):
    url = []
    label = []
    oriLineList = []
    file = open(fileURL, 'r')
    while True:
        line = file.readline()
        if line == '':
            break
        line = line.strip()
        oriLineList.append(line)
        pos = line.find('/')
        label_str = line[:pos]
        curLabel = getLabel(label_str)
        if curLabel == -1:
            raise ValueError('label or url error!')
        image_url = file_root + line
        url.append(image_url)
        label.append(curLabel)
    return url, label, oriLineList


# prepare data module end
# ---------------------------------------------------------




if __name__ == '__main__':
    '''
    trainData, trainLabel = loadImageFromFile(train_file_url)
    testData, testLabel = loadImageFromFile(test_file_url)
    trainData, testData = scaleData(trainData,testData)
    '''
    train_url, train_label = loadImageFromFile(train_file_url)
    for url in train_url:
        loadImage(url)
    print 'check finish!'
