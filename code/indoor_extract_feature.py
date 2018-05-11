# -*- coding:utf-8 -*-

from function_dataBase import *
from function import *

place_mean_file = projectRoot + 'caffemodel/multi-scale-master/places365CNN_mean.binaryproto'
image_mean_file = projectRoot + 'caffemodel/multi-scale-master/imagenet_mean.binaryproto'
# image_finetune_mean_file = projectRoot + 'caffemodel/multi-scale-master/lsq_ImageModel_899_5th_2/mean.binaryproto'

vgg_1prototxt = projectRoot + 'caffemodel/multi-scale-master/deploy_vgg16_1.prototxt'
# vgg_finetune_prototxt = projectRoot + 'caffemodel/multi-scale-master/lsq_ImageModel_899_5th_2/deploy.prototxt'

vgg_place365_model = projectRoot + 'caffemodel/multi-scale-master/vgg16_places365.caffemodel'
vgg_image_model = projectRoot + 'caffemodel/multi-scale-master/VGG_ILSVRC.caffemodel'
# vgg_finetune_model = projectRoot + 'caffemodel/multi-scale-master/lsq_ImageModel_899_5th_2/vgg.caffemodel'

place_mean_data = readmeanfile(place_mean_file)
image_mean_data = readmeanfile(image_mean_file)
# finetune_mean_data = readmeanfile(image_finetune_mean_file)

print "--------------------------init placeCNN---------------------------------"
net_placeCNN = initcaffe(vgg_1prototxt, vgg_place365_model)
net_imageCNN = initcaffe(vgg_1prototxt, vgg_image_model)
# net_finetune = initcaffe(vgg_finetune_prototxt,vgg_finetune_model)

db, cursor = connectdb()


# feature 2 "1+2"论文 baseline
print '-----------------------get feature 2-----------------------------'
get_feature_scala_256(db=db, cursor=cursor, caffenet=net_placeCNN, tbname=traintable,
                      rownum=train_num, datafloder=file_root, mean_data=place_mean_data, featurename='FEATURE2')
get_feature_scala_256(db=db, cursor=cursor, caffenet=net_placeCNN, tbname=testtable,
                      rownum=test_num, datafloder=file_root, mean_data=place_mean_data, featurename='FEATURE2')
print '-----------------------get feature2 finish------------------------'

# feature 3 227尺度
print '-----------------------get feature 3-----------------------------'
get_pool_feature(db=db, cursor=cursor, tbname=traintable, rownum=train_num, picsize=224, cropsize=224,
                 steplength=35, caffenet=net_placeCNN, datafloder=file_root, mean_data=place_mean_data,
                 featurename='FEATURE3')
get_pool_feature(db=db, cursor=cursor, tbname=testtable, rownum=test_num, picsize=224, cropsize=224,
                 steplength=35, caffenet=net_placeCNN, datafloder=file_root, mean_data=place_mean_data,
                 featurename='FEATURE3')
print '-----------------------get feature3 finish------------------------'

# feature 4 451 尺度
print '-----------------------get feature 4-----------------------------'
get_pool_feature(db=db, cursor=cursor, tbname=traintable, rownum=train_num, picsize=451, cropsize=224,
                 steplength=35, caffenet=net_placeCNN, datafloder=file_root, mean_data=place_mean_data,
                 featurename='FEATURE4')
get_pool_feature(db=db, cursor=cursor, tbname=testtable, rownum=test_num, picsize=451, cropsize=224,
                 steplength=35, caffenet=net_placeCNN, datafloder=file_root, mean_data=place_mean_data,
                 featurename='FEATURE4')
print '-----------------------get feature4 finish------------------------'

# feature 5 899 尺度
print '-----------------------get feature 5-----------------------------'
get_pool_feature(db=db, cursor=cursor, tbname=traintable, rownum=train_num, picsize=899, cropsize=224,
                 steplength=70, caffenet=net_imageCNN, datafloder=file_root, mean_data=image_mean_data,
                 featurename='FEATURE5')
get_pool_feature(db=db, cursor=cursor, tbname=testtable, rownum=test_num, picsize=899, cropsize=224,
                 steplength=70, caffenet=net_imageCNN, datafloder=file_root, mean_data=image_mean_data,
                 featurename='FEATURE5')
print '-----------------------get feature5 finish------------------------'

'''
# feature 5 899 尺度
print '-----------------------get feature 5-----------------------------'
get_pool_feature(db=db,cursor=cursor,tbname=traintable,rownum=train_num,picsize=899,cropsize=224,
                 steplength=70,caffenet=net_finetune,datafloder=file_root,mean_data=finetune_mean_data,
                 featurename='FEATURE5')
get_pool_feature(db=db,cursor=cursor,tbname=testtable,rownum=test_num,picsize=899,cropsize=224,
                 steplength=70,caffenet=net_finetune,datafloder=file_root,mean_data=finetune_mean_data,
                 featurename='FEATURE5')
print '-----------------------get feature5 finish------------------------'
'''

'''
# feature 6 : feature 3+4+5 without pca
print '-----------------------get feature 6-----------------------------'
get_feature6(db, cursor)
print '-----------------------get feature6 finish------------------------'
'''
# feature 7 : lsq_cam


'''
# feature 8 : cam feature
print '-----------------------get feature 8-----------------------------'
cam_file_root = '/media/wangchen/newdata1/wangchen/dataSet/Indoor67/CAM/'
cam_train_file = '/media/wangchen/newdata1/wangchen/dataSet/Indoor67/CAM/train_file.txt'
cam_test_file = '/media/wangchen/newdata1/wangchen/dataSet/Indoor67/CAM/test_file.txt'
get_cam_feature(db=db,cursor=cursor,tbname=testtable,file_url=cam_test_file,caffenet=net_placeCNN,
                datafloder=cam_file_root,mean_data=place_mean_data,featurename='FEATURE8')
get_cam_feature(db=db, cursor=cursor, tbname=traintable, file_url=cam_train_file, caffenet=net_placeCNN,
                datafloder=cam_file_root, mean_data=place_mean_data, featurename='FEATURE8')
'''

closedb(db)
