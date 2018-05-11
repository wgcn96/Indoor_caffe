# -*- coding:UTF-8 -*-

projectRoot = '/media/wangchen/newdata1/wangchen/work/Indoor_caffe/'

import sys
import os

try:
    caffe_root = os.environ['CAFFE_ROOT'] + '/'
except KeyError:
    raise KeyError("Define CAFFE_ROOT in ~/.bashrc")

sys.path.insert(1, caffe_root + 'python/')
sys.path.append(projectRoot + 'code/')

import numpy as np
import caffe
import cv2
from function_CAM import *
from function_data import *

num = 10

# net_prototxt = '/media/wangchen/lsq-pc/usr/lsq/indoor/wgcnIndoor/CAM-Python-master/models/deploy_googlenetCAM_places205.prototxt'
# net_weights = '/media/wangchen/lsq-pc/usr/lsq/indoor/wgcnIndoor/CAM-Python-master/models/places_googleletCAM_train_iter_120000.caffemodel'
CAM_image_root = '/media/wangchen/newdata1/wangchen/dataSet/Indoor67/CAM/'

net_prototxt = projectRoot + 'net_prototxt/net_google_indoor_CAM_deploy.prototxt'
net_weights = projectRoot + 'Mymodels/googlenet_indoor/google_indoor_CAM.caffemodel'
mean_file = '/media/wangchen/newdata1/wangchen/dataSet/Indoor67/Images/mean.npy'
layer_last_conv = 'CAM_conv'
layer_fc = 'fc_re'
layer_prob = 'prob'
crop_size = 224

net = caffe.Net(net_prototxt, net_weights, caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))  # c h w
transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
transformer.set_raw_scale('data', 255)
weights_LR = net.params[layer_fc][0].data  # get the softmax layer of the network

url, label, oriLineList = loadImageFromFile(train_file_url)
num_samples = train_num
# shuffle = np.random.permutation(num_samples)  # 用random函数，打乱数据

print 'dealing images...'
count = 0
resultFileList = []
for i in range(num_samples):
    # i = shuffle[i]
    oriLine = oriLineList[i]
    pos = oriLine.find('/')
    imageClass = oriLine[:pos]
    imageName = oriLine[pos + 1:-4]
    imageFolder = imageClass + '/'
    if os.path.exists(CAM_image_root + imageFolder) == False:
        os.mkdir(CAM_image_root + imageFolder)

    curURL = url[i]
    curLabel = label[i]
    curImage = cv2.imread(curURL)
    curImage = im2double(curImage)
    curImageShape = curImage.shape[:2]
    oriImage = curImage
    curImage = cv2.resize(curImage, (crop_size, crop_size))
    curImage = im2double(curImage)
    # cv2.imshow(labeldata[curLabel], curImage)

    net.blobs['data'].data[...][0, :, :, :] = transformer.preprocess('data', curImage)
    out = net.forward()
    scores = out[layer_prob]
    activation_lastconv = net.blobs[layer_last_conv].data

    topNum = 3  # generate heatmap for top X prediction results
    scoresMean = np.mean(scores, axis=0)  # scoresMean.shape is (67,)
    ascending_order = np.argsort(scoresMean)
    IDX_category = ascending_order[::-1]  # [::-1] to sort in descending order

    if IDX_category[0] == curLabel:
        curCAMmapAll = getCAMmap(activation_lastconv, weights_LR[[curLabel], :])
        curCAMmap_crops = curCAMmapAll[:, :, 0]
        count += 1
    else:
        IDX_List = IDX_category[:3]
        curCAMmapAll = getCAMmap(activation_lastconv, weights_LR[IDX_List, :])
        curCAMmap_crops = 0.5 * curCAMmapAll[:, :, 0] + 0.3 * curCAMmapAll[:, :, 1] + 0.2 * curCAMmapAll[:, :, 2]

    curCAMmapLarge_crops = cv2.resize(curCAMmap_crops, (curImageShape[1], curImageShape[0]))
    # threshold = get_threshold(curCAMmapLarge_crops, 0.2)
    # x1, y1, x2, y2 = get_bbox(curCAMmapLarge_crops, threshold)
    x1, y1, x2, y2 = auto_adap_bbox(img=curCAMmapLarge_crops, thresholdRatio=0.2, coverAreaRatio=0.8, stepSize=0.1)
    curHeatMap = im2double(curCAMmapLarge_crops)
    curHeatMap = py_map2jpg(curHeatMap, None, 'jet')  # 使用热力图显示
    # cv2.imwrite(projectRoot + 'Mytest/img1_act.jpg', curHeatMap)

    curResult = oriImage * 0.3 + im2double(curHeatMap) * 0.7
    cv2.rectangle(curResult, (y1, x1), (y2, x2), (0, 0.99, 0), 2)
    curResult = im2double(curResult)
    # cv2.imwrite(projectRoot + 'Mytest/img1_act_bbox.jpg', curResult)  # 添加 bounding box
    # cv2.imshow(str(i) + '_result', curResult)

    CAM_crop = oriImage[x1:x2, y1:y2, :]
    CAM_crop = im2double(CAM_crop)
    CAM_crop *= 255
    # cv2.imshow(str(i) + '_crop', CAM_crop)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    newImageURL = CAM_image_root + imageFolder + imageName + '_CAM.jpg'
    writeLine = imageFolder + imageName + '_CAM.jpg\n'
    resultFileList.append(writeLine)
    cv2.imwrite(newImageURL, CAM_crop)

fileURL = CAM_image_root + 'train_file.txt'
file = open(fileURL, 'w')
file.writelines(resultFileList)
file.close()
print 'finish'
