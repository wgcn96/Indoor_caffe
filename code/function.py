# -*- coding:utf-8 -*-

import os
import sys

projectRoot = '/media/wangchen/newdata1/wangchen/work/Indoor_caffe/'
caffePath = '/home/wangchen/caffe'
os.environ['CAFFE_ROOT'] = caffePath

try:
    caffe_root = os.environ['CAFFE_ROOT'] + '/'
    print caffe_root
except KeyError:
    raise KeyError("Define CAFFE_ROOT in ~/.bashrc")
sys.path.insert(1, caffe_root + 'python/')
sys.path.append \
    (projectRoot + 'code/')

import time
import cv2
import caffe
import numpy as np

from sklearn import svm
from sklearn import metrics
import xlwt
from sklearn import preprocessing
from sklearn.decomposition import PCA
import random

from function_dataBase import *


# =========================caffe接口======================
# ========================================================

def initcaffe(protourl, modelurl):
    caffe.set_device(1)
    caffe.set_mode_gpu()
    net = caffe.Net(protourl, modelurl, caffe.TEST)
    return net


def imageTransformer(net, mean_data):
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))  # c h w
    mean = mean_data.mean(1).mean(1)
    transformer.set_mean('data', mean)
    # transformer.set_raw_scale('data', 255)
    return transformer


# =======================模型均值===========================
# readmeanfile : return mean data (w h c)
# makeplaces205meandata: 生成pl205数据集的meanfile
# ====================================================
def readmeanfile(meanfile):
    mean_blob = caffe.proto.caffe_pb2.BlobProto()
    mean_blob.ParseFromString(open(meanfile, 'rb').read())
    # 将均值blob转为np.array
    mean_npy = caffe.io.blobproto_to_array(mean_blob)
    return mean_npy[0]
    # return mean_npy[0][:, 16:240, 16:240]


def makeplaces205meandata():
    out = np.zeros([3, 224, 224])
    out[0, :, :] = 105.487823486
    out[1, :, :] = 113.741088867
    out[2, :, :] = 116.060394287
    return out


# =================特征计算===============================================================
# calture_normal_feature: 对于一张图片，计算256尺寸下 center crop特征 (1+2 论文中的baseline)
# calture_pool_feature:   对于一张图片，计算设定尺寸下 以一定步长截取部分图片的特征 然后maxpool
# get_feature_scala_256:  计算整个数据库normal_feature
# get_pool_feature：      计算整个数据库pool_feature
# get_cam_feature:
# ========================================================================================
def calture_normal_feature(pic, caffenet, mean_data):

    transformer = imageTransformer(caffenet, mean_data)

    feature = np.zeros((1, 4096))
    # cv2.imshow('0',pic)
    sp = pic.shape
    y = sp[0]
    x = sp[1]
    if x < y:
        y = (int)((y * 256) / x)
        x = 256
    else:
        x = (int)((x * 256) / y)
        y = 256

    im_256 = cv2.resize(pic, (x, y))
    # cv2.imshow('1',im_256)
    # cv2.waitKey(0)
    im_224 = im_256[((int)(y / 2) - 112):((int)(y / 2) + 112), ((int)(x / 2) - 112):((int)(x / 2) + 112)]


    im = transformer.preprocess('data', im_224)

    '''
    im = np.transpose(im_224, (2, 0, 1))
    im = im - mean_data
    '''
    im = np.resize(im, (1, 3, 224, 224))

    caffenet.blobs['data'].data[...] = im
    caffenet.forward()
    feature[0] = caffenet.blobs['fc7'].data[0]

    return feature


def calture_pool_feature(pic, picsize, cropsize, steplength, caffenet, mean_data, parallelnum=1, dropLabel=False):
    im = np.zeros((parallelnum, 3, cropsize, cropsize))
    transformer = imageTransformer(caffenet, mean_data)

    feature_max = np.zeros((parallelnum, 4096))
    feature_max = feature_max - 9999
    # 对每个尺度框图并提取特征
    step = (picsize - cropsize) // steplength
    for m in range(step + 1):
        for n in range(step + 1):
            x = m * steplength
            y = n * steplength
            if x > picsize - cropsize:
                x = picsize - cropsize
            if y > picsize - cropsize:
                y = picsize - cropsize
            crop = pic[:, y:y + cropsize, x:x + cropsize, :]  ### crop是四维的数组 n h w c
            if dropLabel == True:
                dropInt = random.randint(1, 100)
                if dropInt > 75:
                    continue

            for i in range(parallelnum):
                im[i] = transformer.preprocess('data', crop[i])
            '''

            im = np.transpose(crop, (0, 3, 1, 2))
            im = im - mean_data
            '''
            caffenet.blobs['data'].data[...] = im
            caffenet.forward()
            tmp = caffenet.blobs['fc7'].data

            for i in range(parallelnum):
                for j in range(4096):
                    if tmp[i][j] >= feature_max[i][j]:
                        feature_max[i][j] = tmp[i][j]
                        # tmp[i][j] = tmp[i][j]/(step+1)*(step+1)
                        # feature_mean[i][j] = feature_mean[i][j] + tmp[i][j]
    return feature_max


def get_feature_scala_256(db, cursor, caffenet, tbname, rownum, datafloder, mean_data, featurename):
    feature_all = []
    for i in range(rownum):
        print '============current id is :%d ==============' % (i + 1)
        sql = "SELECT URL FROM " + tbname + " WHERE ID = '%d'" % (i + 1)
        cursor.execute(sql)
        result = cursor.fetchall()
        url = datafloder + result[0][0]
        im_ori = cv2.imread(url)
        cur_feature = calture_normal_feature(im_ori, caffenet, mean_data)
        feature_all.extend(cur_feature)
    feature_all = np.asarray(feature_all, dtype='float32')
    print feature_all.shape
    # 写入数据库
    write_feature_to_db(db=db, cursor=cursor, table_name=tbname, featurename=featurename, feature=feature_all)


def get_pool_feature(db, cursor, tbname, rownum, picsize, cropsize, steplength, caffenet, datafloder, mean_data,
                     featurename, parallelnum=1):
    feature_max = []
    for i in range(int(rownum / parallelnum)):
        print '============current id is :%d ==============' % (i * parallelnum + 1)
        sql = "SELECT URL FROM " + tbname + " WHERE ID >= '%d' and ID <= '%d'" % (
            i * parallelnum + 1, (i + 1) * parallelnum)
        cursor.execute(sql)
        result = cursor.fetchall()

        im = np.zeros((parallelnum, picsize, picsize, 3))
        for j in range(parallelnum):
            url = datafloder + result[j][0]
            im_ori = cv2.imread(url)
            im[j, :, :, :] = cv2.resize(im_ori, (picsize, picsize))

        current_max = calture_pool_feature(im, picsize, cropsize, steplength, caffenet, mean_data, parallelnum)
        feature_max.extend(current_max)
    feature_max = np.array(feature_max, dtype='float32')
    print feature_max.shape
    write_feature_to_db(db=db, cursor=cursor, table_name=tbname, featurename=featurename, feature=feature_max)


def get_cam_feature(db, cursor, tbname, file_url, caffenet, datafloder, mean_data, featurename):

    transformer = imageTransformer(caffenet, mean_data)

    feature_max = []
    file = open(file_url, 'r')
    count = 0
    while True:
        current_max = np.zeros((1, 4096))
        current_max -= 9999
        line = file.readline()
        line = line.strip()
        if not line:
            break
        count += 1
        print '----------------------current ID is: {}---------------------'.format(count)
        url = datafloder + line
        img = cv2.imread(url)
        x = float(img.shape[0])
        y = float(img.shape[1])

        if x < 224 or y < 224:
            scale1 = x / y
            scale2 = y / x
            if scale1 < scale2:
                img = cv2.resize(img, (int(scale2 * 224), 224))
            else:
                img = cv2.resize(img, (224, int(scale1 * 224)))

        x = img.shape[0]
        y = img.shape[1]

        if x > 451 and y > 451:
            steplength = 70
        else:
            steplength = 35

        step_x = (x - 224) / steplength + 1
        step_y = (y - 224) / steplength + 1

        for i in range(step_x):
            for j in range(step_y):
                x = i * steplength
                y = j * steplength
                crop = img[x:x + 224, y:y + 224, :]

                
                im = transformer.preprocess('data', crop)

                '''
                im = np.transpose(crop, (2, 0, 1))
                im = im - mean_data
                '''
                im = np.resize(im, (1, 3, 224, 224))
                caffenet.blobs['data'].data[...] = im
                caffenet.forward()
                tmp = caffenet.blobs['fc7'].data

                for k in range(4096):
                    if tmp[0][k] >= current_max[0][k]:
                        current_max[0][k] = tmp[0][k]
        feature_max.extend(current_max)
    feature_max = np.array(feature_max, dtype='float32')
    print feature_max.shape
    file.close()
    write_feature_to_db(db=db, cursor=cursor, table_name=tbname, featurename=featurename, feature=feature_max)


# ======================提取特征============================
# ==========================================================
# feature6: “1+2”的feature PCA
def get_feature6(db, cursor):
    FEATURE3_train_data, train_label = read_feature(db, cursor, table_name=traintable, featurename="FEATURE3",
                                                    num=train_num)
    FEATURE3_test_data, test_label = read_feature(db, cursor, table_name=testtable, featurename="FEATURE3",
                                                  num=test_num)
    FEATURE4_train_data, train_label = read_feature(db, cursor, table_name=traintable, featurename="FEATURE4",
                                                    num=train_num)
    FEATURE4_test_data, test_label = read_feature(db, cursor, table_name=testtable, featurename="FEATURE4",
                                                  num=test_num)
    FEATURE5_train_data, train_label = read_feature(db, cursor, table_name=traintable, featurename="FEATURE5",
                                                    num=train_num)
    FEATURE5_test_data, test_label = read_feature(db, cursor, table_name=testtable, featurename="FEATURE5",
                                                  num=test_num)

    FEATURE6_train_data = np.concatenate((FEATURE3_train_data, FEATURE4_train_data, FEATURE5_train_data), 1)
    FEATURE6_test_data = np.concatenate((FEATURE3_test_data, FEATURE4_test_data, FEATURE5_test_data), 1)

    # FEATURE6_train_data,FEATURE6_test_data = myPCA(pre_6_train_data,pre_6_test_data)

    write_feature_to_db(db, cursor, table_name=traintable, featurename='FEATURE6', feature=FEATURE6_train_data)
    write_feature_to_db(db, cursor, table_name=testtable, featurename='FEATURE6', feature=FEATURE6_test_data)


# ============================SVM 与 PCA==============================
# ====================================================================
def myPCA(train_feature, test_feature, component=0.99):
    pca = PCA(n_components=component)
    scaler = preprocessing.StandardScaler().fit(train_feature)
    train_feature_scale = scaler.transform(train_feature)
    test_feature_scale = scaler.transform(test_feature)
    pca.fit(train_feature_scale)
    train_feature_pca = pca.transform(train_feature_scale)
    test_feature_pca = pca.transform(test_feature_scale)
    return train_feature_pca, test_feature_pca


def mySVM(train_data, test_data, train_label, test_label):
    clf = svm.SVC(kernel='linear')
    clf.fit(train_data, train_label)
    y_hat = clf.predict(test_data)
    # m_accuracy = metrics.accuracy_score(test_label, y_hat)
    # m_f1score = metrics.f1_score(y_true=test_label,y_pred=y_hat,average='macro')
    m_precision = metrics.precision_score(test_label, y_hat, average='macro')
    return m_precision  # , m_accuracy



def fsvm(db, cursor, featurename):
    print '从数据库取' + featurename + '数据...'
    train_data, train_label = read_feature(db, cursor, traintable, featurename, train_num)
    test_data, test_label = read_feature(db, cursor, testtable, featurename, test_num)
    print '训练SVM并测试...'
    result = mySVM(train_data, test_data, train_label, test_label)
    print 'the result is : {}'.format(result)


# ===============结果计算和输出部分=================
# calculate_result 计算正确率 召回率等
# calculate_detail 统计所有错误分类
# detailtofile     将错误分类 URL ID reallabel pre打印到文件
# matrixtoexcl     打印错误分类混淆矩阵
# =================================================
def detail_result(test_label, y_hat):
    m_accuracy = metrics.accuracy_score(test_label, y_hat)
    m_recall = metrics.recall_score(test_label, y_hat, average='macro')
    m_f1score = metrics.f1_score(test_label, y_hat, average='macro')
    return m_accuracy, m_recall, m_f1score


def confusionMatrix(test_label, y_hat):
    detailList = []
    matrixTabel = np.zeros([67, 67])
    for i in range(len(test_label)):
        # if y_hat[i] != test_label[i]:
        tmp = [i + 1, test_label[i], y_hat[i]]
        detailList.append(tmp)
        matrixTabel[test_label[i], y_hat[i]] += 1
    detailList = np.asarray(detailList, dtype='int32')
    return matrixTabel, detailList


def detailToFile(numpy_data, outfileurl):
    # np.savetxt(outfileurl, numpy_data, fmt='%2.4f')
    np.savetxt(outfileurl, numpy_data, fmt='%d')


def matrixToExcel(matrixTable, xlsurl):
    wbk = xlwt.Workbook()
    sheet = wbk.add_sheet('sheet 1')
    for i in range(67):
        sheet.write(0, i + 1, labeldata[i])
        sheet.write(i + 1, 0, labeldata[i])
    for i in range(len(matrixTable)):
        for j in range(len(matrixTable)):
            if matrixTable[i, j] != 0:
                sheet.write(j + 1, i + 1, matrixTable[i, j])
    wbk.save(xlsurl)


def writeInfo(fileURL, info):
    with open(fileURL, 'a') as f:
        f.write(info)
    f.close()


def validationImage(db, cursor, tbname, rownum):
    for i in range(rownum):
        print '============current id is :%d ==============\r' % (i + 1),
        sql = "SELECT URL FROM " + tbname + " WHERE ID = '%d'" % (i + 1)
        cursor.execute(sql)
        result = cursor.fetchall()
        url = result[0][0]
        url = file_root + url
        im_ori = cv2.imread(url)
        sp = im_ori.shape


if __name__ == '__main__':
    print 'begin validation...'
    db, cursor = connectdb()
    validationImage(db, cursor, "indoor67train", 5360)
    validationImage(db, cursor, "indoor67test", 1340)
    print 'finish!'
