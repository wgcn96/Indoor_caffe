# -*- coding: UTF-8 -*-

import sys
import os

try:
    caffe_root = os.environ['CAFFE_ROOT'] + '/'
except KeyError:
    raise KeyError("Define CAFFE_ROOT in ~/.bashrc")

sys.path.insert(1, caffe_root + 'python/')

import caffe
from function_dataBase import *

caffe.set_device(1)
caffe.set_mode_gpu()


def im2double(im):
    return cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)


def py_map2jpg(imgmap, rang, colorMap):
    if rang is None:
        rang = [np.min(imgmap), np.max(imgmap)]

    heatmap_x = np.round(imgmap * 255).astype(np.uint8)

    return cv2.applyColorMap(heatmap_x, cv2.COLORMAP_JET)


def showcam(net, fileurl, mode=1):
    out_layer = 'CAM_fc'
    crop_size = 224
    last_conv = 'CAM_conv'

    weights_LR = net.params[out_layer][0].data  # get the softmax layer of the network
    image = cv2.imread(fileurl)
    while image is None:
        print fileurl
        print "try to read again"
        image = cv2.imread(fileurl)
    image = cv2.resize(image, (256, 256))
    # Take center crop.
    center = np.array(image.shape[:2]) / 2.0
    crop = np.tile(center, (1, 2))[0] + np.concatenate([
        -np.array([crop_size, crop_size]) / 2.0,
        np.array([crop_size, crop_size]) / 2.0
    ])
    crop = crop.astype(int)
    input_ = image[crop[0]:crop[2], crop[1]:crop[3], :]

    mean_data = np.zeros([3, 224, 224])
    mean_data[0, :, :] = 105.487823486
    mean_data[1, :, :] = 113.741088867
    mean_data[2, :, :] = 116.060394287

    im = np.transpose(input_, (2, 0, 1))
    im = im - mean_data
    net.blobs['data'].data[...] = im
    out = net.forward()
    scores = out['prob']
    activation_lastconv = net.blobs[last_conv].data

    ## Class Activation Mapping
    topNum = 3  # generate heatmap for top X prediction results
    scoresMean = np.mean(scores, axis=0)
    ascending_order = np.argsort(scoresMean)
    IDX_category = ascending_order[::-1]  # [::-1] to sort in descending order
    curCAMmapAll = py_returnCAMmap(activation_lastconv, weights_LR[IDX_category[:topNum], :])
    curResult = im2double(image)

    # for one image
    curCAMmap_crops1 = curCAMmapAll[:, :, 0]
    curCAMmap_crops2 = curCAMmapAll[:, :, 1]
    curCAMmap_crops3 = curCAMmapAll[:, :, 2]
    """
	for i in range(topNum):
		testcurCAM = curCAMmapAll[:,:,i]
		tmpImg = im2double(testcurCAM)
		cv2.imshow(str(i),tmpImg)
		cv2.waitKey(0)
	"""

    if mode == 1:
        curCAMmap_crops = curCAMmap_crops1 * 0.5 + curCAMmap_crops2 * 0.3 + curCAMmap_crops3 * 0.2
    elif mode == 2:
        curCAMmap_crops = curCAMmap_crops1 * scoresMean[IDX_category[0]] + curCAMmap_crops2 * scoresMean[
            IDX_category[1]] + curCAMmap_crops3 * scoresMean[IDX_category[2]]
    elif mode == 3:
        curCAMmap_crops = curCAMmap_crops1
    elif mode == 4:
        curCAMmap_crops = curCAMmap_crops1 * 0.333 + curCAMmap_crops2 * 0.333 + curCAMmap_crops3 * 0.333
    elif mode == 5:
        if scoresMean[IDX_category[0]] > 0.6:
            scoresMean[IDX_category[0]] = 0.6
        curCAMmap_crops = curCAMmap_crops1 * scoresMean[IDX_category[0]] + curCAMmap_crops2 * scoresMean[
            IDX_category[1]] + curCAMmap_crops3 * scoresMean[IDX_category[2]]

    curCAMmapLarge_crops = cv2.resize(curCAMmap_crops, (512, 512))

    curHeatMap = cv2.resize(im2double(curCAMmapLarge_crops), (512, 512))  # this line is not doing much

    # cv2.imshow(str(5), curHeatMap)
    # cv2.waitKey(0)

    curHeatMap = im2double(curHeatMap)
    cammap = np.round(curHeatMap * 255).astype(np.uint8)

    # cv2.imshow(str(6), cammap)
    # cv2.waitKey(0)

    curHeatMap = py_map2jpg(curHeatMap, None, 'jet')
    image = cv2.resize(image, (512, 512))

    # cv2.imshow(str(7), curHeatMap)
    # cv2.waitKey(0)

    # cv2.imwrite("./heatmap.jpg", curHeatMap)
    curHeatMap = im2double(image) * 0.3 + im2double(curHeatMap) * 0.7

    # cv2.imshow("123", curHeatMap)
    # cv2.waitKey(0)
    # print curCAMmap_crops
    return cammap


def lsq_calculateKeyRegion(cammap, cropsize, url, steplength, caffenet, meanfile, countmin):
    feature_max = [-9999 for x in range(4096)]
    # 3 对每个尺度框图并提取特征
    picsize = 512
    im_ori = cv2.imread(url)
    im_ori = cv2.resize(im_ori, (picsize, picsize))
    step = (512 - cropsize) // steplength
    cammap = np.array(cammap)
    Max = max(cammap.flatten())
    count = 0
    shreshold = 0
    while count < countmin:
        count = 0
        for m in range(step + 1):
            for n in range(step + 1):
                x = m * steplength
                y = n * steplength
                if x > 512 - cropsize:
                    x = picsize - cropsize
                if y > picsize - cropsize:
                    y = picsize - cropsize
                if cammap[x + 112, y + 112] > Max - shreshold:
                    count = count + 1
        if count < countmin:
            shreshold = shreshold + 5
            if shreshold >= 255:
                shreshold = 255
                break;
    print count
    for m in range(step + 1):
        for n in range(step + 1):
            x = m * steplength
            y = n * steplength
            if x > 512 - cropsize:
                x = picsize - cropsize
            if y > picsize - cropsize:
                y = picsize - cropsize
            if cammap[x + 112, y + 112] > Max - shreshold:
                crop = im_ori[y:y + cropsize, x:x + cropsize]
                im = np.transpose(crop, (2, 0, 1))
                im = im - meanfile
                caffenet.blobs['data'].data[...] = im
                caffenet.forward()
                tmp = caffenet.blobs['fc7'].data[0]

                # 4 将特征pooling
                for j in range(4096):
                    if tmp[j] >= feature_max[j]:
                        feature_max[j] = tmp[j]
    return feature_max


def py_returnCAMmap(activation, weights_LR):
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


def lsq_get_cam_feature(db, cursor, tbname, rownum, net_cam, net_feature, mean_data, datafloder, featurename,
                        steplength, countmin, mode=1):
    feature_max = []
    for i in range(rownum):
        print '============current id is :%d ==============\r' % (i + 1),
        sql = "SELECT URL FROM " + tbname + " WHERE ID = '%d'" % (i + 1)
        cursor.execute(sql)
        result = cursor.fetchall()
        url = datafloder + result[0][0]
        cammap = showcam(net_cam, url, mode)
        tmp = lsq_calculateKeyRegion(cammap, 224, url, steplength, net_feature, mean_data, countmin)
        feature_max.append(tmp)

    feature_max = np.asarray(feature_max, dtype='float32')
    print feature_max.shape
    write_feature_to_db(db=db, cursor=cursor, table_name=tbname, featurename=featurename, feature=feature_max)


if __name__ == '__main__':
    projectRoot = '/media/wangchen/newdata1/wangchen/work/Indoor_caffe/'
    CAM_net_prototxt = projectRoot + 'caffemodel/cam-master/deploy_googlenetCAM_places205.prototxt'
    CAM_net_weights = projectRoot + 'caffemodel/cam-master/places_googleletCAM_train_iter_120000.caffemodel'
    vgg_1prototxt = projectRoot + 'caffemodel/multi-scale-master/deploy_vgg16_1.prototxt'
    vgg_place365_model = projectRoot + 'caffemodel/multi-scale-master/vgg16_places365.caffemodel'
    file_root = '/media/wangchen/newdata1/wangchen/dataSet/Indoor67/Images/'
    mean_file = '/media/wangchen/newdata1/wangchen/dataSet/Indoor67/Images/mean.binaryproto'
    blob = caffe.proto.caffe_pb2.BlobProto()
    bin_mean = open(mean_file, 'rb').read()
    blob.ParseFromString(bin_mean)
    arr = np.array(caffe.io.blobproto_to_array(blob))
    mean_data = arr[0]
    CAM_net = caffe.Net(CAM_net_prototxt, CAM_net_weights, caffe.TEST)
    vgg_net = caffe.Net(vgg_1prototxt, vgg_place365_model, caffe.TEST)
    db, cursor = connectdb()
    lsq_get_cam_feature(db, cursor, tbname=testtable, rownum=test_num, net_cam=CAM_net, net_feature=vgg_net,
                        mean_data=mean_data, datafloder=file_root, featurename='FEATURE7', steplength=35,
                        countmin=50)
    lsq_get_cam_feature(db, cursor, tbname=traintable, rownum=train_num, net_cam=CAM_net, net_feature=vgg_net,
                        mean_data=mean_data, datafloder=file_root, featurename='FEATURE7', steplength=35,
                        countmin=50)

    closedb(db)
