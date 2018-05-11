#!/usr/bin/env bash
~/caffe/build/examples/cpp_classification/classification.bin \
   /media/wangchen/newdata1/wangchen/work/Indoor_caffe/net_prototxt/net_google_indoor_deploy.prototxt \
   /media/wangchen/newdata1/wangchen/work/Indoor_caffe/Mymodels/googlenet_indoor/google_indoor_bestmodel.caffemodel \
   /media/wangchen/newdata1/wangchen/dataSet/Indoor67/Images/mean.binaryproto \
   /media/wangchen/newdata1/wangchen/dataSet/Indoor67/Images/label.txt \
   /media/wangchen/newdata1/wangchen/dataSet/Indoor67/Images/kitchen/int474.jpg
