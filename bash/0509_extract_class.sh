#!/usr/bin/env bash

WORKFOLDER=/media/wangchen/newdata1/wangchen/work/Indoor_caffe/code

python $WORKFOLDER/indoor_extract_feature.py
python $WORKFOLDER/indoor_classification.py