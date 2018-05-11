

import sys
import os

try:
	caffe_root = os.environ['CAFFE_ROOT'] + '/'
except KeyError:
  	raise KeyError("Define CAFFE_ROOT in ~/.bashrc")

sys.path.insert(1, caffe_root+'python/')
import numpy as np
import caffe
print caffe.__path__

np.set_printoptions(threshold=np.NaN)

mean_file = '/media/wangchen/newdata1/wangchen/work/Indoor_caffe/caffemodel/multi-scale-master/imagenet_mean.binaryproto'
save_file = '/media/wangchen/newdata1/wangchen/work/Indoor_caffe/caffemodel/places365-master/places365CNN_mean.npy'
blob = caffe.proto.caffe_pb2.BlobProto()
bin_mean = open( mean_file , 'rb' ).read()
blob.ParseFromString(bin_mean)
arr = np.array( caffe.io.blobproto_to_array(blob) )
npy_mean = arr[0]
# tmp = npy_mean.mean(1)
# tmp = tmp.mean(1)
# img = np.ones((3,224,224))
# img *= 255
# img[0] -= tmp[0]
# img[1] -= tmp[1]
# img[2] -= tmp[2]
# print img
# np.savetxt('current.txt',img)
np.save( save_file , npy_mean )