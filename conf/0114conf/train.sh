#!/usr/bin/env sh

~/caffe/build/tools/caffe train -solver /media/wangchen/newdata1/wangchen/work/Indoor_caffe/conf/0114conf/solver.prototxt -weights /media/wangchen/newdata1/wangchen/work/Indoor_caffe/caffemodel/places365-master/googlenet_places365.caffemodel >& /media/wangchen/newdata1/wangchen/work/Indoor_caffe/result/0114result.log &

tail -f /media/wangchen/newdata1/wangchen/work/Indoor_caffe/result/0114result.log
