"""
Resnet Test
Get Resnet feature
Author: Kaihua Tang
"""

import math
import time
import tensorflow as tf
import densenet121 as resnet
import numpy as np
import scipy.io as scio
from scipy import misc
from utils import *
from configs import configs
import matplotlib.pyplot as plt
from read_data import read_data
feature_path = "./resnet_feature.mat"



#Lists that store name of image and its label


"61114 is 801 - 2000"
"162830 is 1991 - 2000"
"150000 is 18XX - 2000"
"14420 is 1 - 200"
read = read_data(root_dir = configs['root_dir'], save_dir = configs['save_dir'], image_size = configs['image_size'], per = configs['per'])
if configs['saved_npy'] == False:
    read.save_data()
read.build_train_input()
read.build_test_input()

res_feature = []

with tf.Session() as sess:

    images = tf.placeholder(tf.float32, shape = [1, configs['image_size'], configs['image_size'], configs['channel']])
    train_mode = tf.placeholder(tf.bool)

    # build resnet model
    resnet_model = resnet.ResNet(ResNet_npy_path = configs['model_path'])
    resnet_model.build(images, configs['num_classes'], train_mode)

    sess.run(tf.global_variables_initializer())
    resnet_model.set_is_training(False)

    for i in range(14420):
        if(i%1000 == 0):
            print(i)
        zz = sess.run(resnet_model.y_soft, feed_dict={images: np.expand_dims(read.test_images[i],axis = 0), train_mode: False})
        plt.imshow(zz[0][:,:,1])
        print(zz[0][:,:,1])
        plt.pause(0.1)
    	# res_feature.append(fc1[0])

    # scio.savemat(feature_path,{'feature' : res_feature})
