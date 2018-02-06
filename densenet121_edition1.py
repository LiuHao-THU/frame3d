"""
A Trainable ResNet Class is defined in this file
Author: LIUHao
"""
"""
dense net name 
conv1/bn + conv1/scale + conv1//relu + conv1 + pool2(max)
conv2_1/x1.2(/bn+/scale+relu+conv)
...
conv2_6/x1.2(/bn+/scale+relu+conv) concat
conv2_blk/bn + conv2_blk/scale + conv2_blk/relu + conv2_blk + pool2(avg)
conv3_1/x1.2(/bn+/scale+relu+conv)
...
conv3_12/x1.2(/bn+/scale+relu+conv) concat
conv3_blk/bn + conv3_blk/scale + conv3_blk/relu + conv3_blk + pool3(avg)
conv4_1/x1.2(/bn+/scale+relu+conv)
...
conv4_2/x1.2(/bn+/scale+relu+conv) concat
conv4_blk/bn + conv4_blk/scale + conv4_blk/relu + conv4_blk + pool4(avg)
conv5_1/x1.2(/bn+/scale+relu+conv)
...
conv5_16/x1.2(/bn+/scale+relu+conv) concat
"""
# input dimension batch_size * 224 * 224 * channels
# out_size batch_size * 8 * 8 * 1024
import math
import numpy as np
import tensorflow as tf
from functools import reduce
import six
from configs import configs
K = 32
middle_layer = K * 4

class ResNet:
	# some properties
    """
    Initialize function
    """
    def __init__(self, ResNet_npy_path=None, trainable=True, open_tensorboard=False, dropout=0.8):
        if ResNet_npy_path is not None:
            self.data_dict = np.load(ResNet_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.open_tensorboard = open_tensorboard
        self.dropout = dropout
        self.is_training = True
        # self.train_phase = tf.Variable()
    def set_is_training(self, isTrain):
    	self.is_training = isTrain

    def build(self, rgb, label_num, train_mode=None, last_layer_type = "softmax"):
        """
        load variable from npy to build the Resnet or Generate a new one
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """
        self.train_mode = train_mode
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - configs['VGG_MEAN'][0],
            green - configs['VGG_MEAN'][1],
            red - configs['VGG_MEAN'][2],
        ])
        print(bgr.get_shape().as_list())
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        #stage2
        # self.conv1 = self.conv_layer(bottom = bgr, kernel_size = 7, in_channels = 3, out_channels = 64, stride = 2, name = "conv1")# 112*112     
        # self.bn_1 = self.batch_norm_layer(name = 'conv1', bottom = self.conv1, phase_train = self.train_mode)
        # self.relu_1 = tf.nn.relu(self.bn_1)
        self.conv1 = self.Conv_Bn_Relu(name = 'conv1', bottom = bgr, output_channels = 64, kernel_size = 7, stride = 2)
        self.convd = self.Conv_Bn_Relu(name = 'convd', bottom = bgr, output_channels = 64, kernel_size = 7, stride = 1)
        self.pool1 = self.max_pool(bottom = self.conv1, kernel_size = 3, stride = 2, name = "pool1")# 56*56* 64

        self.block1_1 = self.Dense_Block(self.pool1, name = "conv2_1", stride = 1)# 56*56*256
        self.block1_2 = self.Dense_Block(self.block1_1, name = "conv2_2", stride = 1)# 56*56*256
        self.block1_3 = self.Dense_Block(self.block1_2, name = "conv2_3", stride = 1)# 56*56*256
        self.block1_4 = self.Dense_Block(self.block1_3, name = "conv2_4", stride = 1)# 56*56*256
        self.block1_5 = self.Dense_Block(self.block1_4, name = "conv2_5", stride = 1)# 56*56*256
        self.block1_6 = self.Dense_Block(self.block1_5, name = "conv2_6", stride = 1)# 56*56*256
        #stage3
        self. blk2 = self.BN_Relu_Conv("conv2_blk", self.block1_6, input_channels = self.block1_6.get_shape().as_list()[-1], 
        								output_channels =self.block1_6.get_shape().as_list()[-1]//2, kernel_size = 1, stride = 1)

        self.pool2 = self.avg_pool(bottom = self. blk2, kernel_size = 2, stride = 2, name = "pool2")
        self.block2_1 = self.Dense_Block(self.pool2, name = "conv3_1", stride = 1)# 28*28*512
        self.block2_2 = self.Dense_Block(self.block2_1, name = "conv3_2", stride = 1)# 28*28*512
        self.block2_3 = self.Dense_Block(self.block2_2, name = "conv3_3", stride = 1)# 28*28*512
        self.block2_4 = self.Dense_Block(self.block2_3, name = "conv3_4", stride = 1)# 28*28*512
        self.block2_5 = self.Dense_Block(self.block2_4, name = "conv3_5", stride = 1)# 28*28*512
        self.block2_6 = self.Dense_Block(self.block2_5, name = "conv3_6", stride = 1)# 28*28*512
        self.block2_7 = self.Dense_Block(self.block2_6, name = "conv3_7", stride = 1)# 28*28*512
        self.block2_8 = self.Dense_Block(self.block2_7, name = "conv3_8", stride = 1)# 28*28*512
        self.block2_9 = self.Dense_Block(self.block2_8, name = "conv3_9", stride = 1)# 28*28*512
        self.block2_10 = self.Dense_Block(self.block2_9, name = "conv3_10", stride = 1)# 28*28*512
        self.block2_11 = self.Dense_Block(self.block2_10, name = "conv3_11", stride = 1)# 28*28*512
        self.block2_12 = self.Dense_Block(self.block2_11, name = "conv3_12", stride = 1)# 28*28*512
        self. blk3 = self.BN_Relu_Conv("conv3_blk", self.block2_12, input_channels = self.block2_12.get_shape().as_list()[-1], 
        								output_channels =self.block2_12.get_shape().as_list()[-1]//2, kernel_size = 1, stride = 1)
        self.pool3 = self.avg_pool(bottom = self. blk3, kernel_size = 2, stride = 2, name = "pool3")
        #stage4
        self.block3_1 = self.Dense_Block(self.pool3, name = "conv4_1", stride = 1)# 14*14*1024
        self.block3_2 = self.Dense_Block(self.block3_1, name = "conv4_2", stride = 1)# 14*14*1024
        self.block3_3 = self.Dense_Block(self.block3_2, name = "conv4_3", stride = 1)# 14*14*1024
        self.block3_4 = self.Dense_Block(self.block3_3, name = "conv4_4", stride = 1)# 14*14*1024
        self.block3_5 = self.Dense_Block(self.block3_4, name = "conv4_5", stride = 1)# 14*14*1024
        self.block3_6 = self.Dense_Block(self.block3_5, name = "conv4_6", stride = 1)# 14*14*1024
        self.block3_7 = self.Dense_Block(self.block3_6, name = "conv4_7", stride = 1)# 14*14*1024
        self.block3_8 = self.Dense_Block(self.block3_7, name = "conv4_8", stride = 1)# 14*14*1024
        self.block3_9 = self.Dense_Block(self.block3_8, name = "conv4_9", stride = 1)# 14*14*1024
        self.block3_10 = self.Dense_Block(self.block3_9, name = "conv4_10", stride = 1)# 14*14*1024
        self.block3_11 = self.Dense_Block(self.block3_10, name = "conv4_11", stride = 1)# 14*14*1024
        self.block3_12 = self.Dense_Block(self.block3_11, name = "conv4_12", stride = 1)# 14*14*1024
        self.block3_13 = self.Dense_Block(self.block3_12, name = "conv4_13", stride = 1)# 14*14*1024
        self.block3_14 = self.Dense_Block(self.block3_13, name = "conv4_14", stride = 1)# 14*14*1024
        self.block3_15 = self.Dense_Block(self.block3_14, name = "conv4_15", stride = 1)# 14*14*1024
        self.block3_16 = self.Dense_Block(self.block3_15, name = "conv4_16", stride = 1)# 14*14*1024
        self.block3_17 = self.Dense_Block(self.block3_16, name = "conv4_17", stride = 1)# 14*14*1024
        self.block3_18 = self.Dense_Block(self.block3_17, name = "conv4_18", stride = 1)# 14*14*1024
        self.block3_19 = self.Dense_Block(self.block3_18, name = "conv4_19", stride = 1)# 14*14*1024
        self.block3_20 = self.Dense_Block(self.block3_19, name = "conv4_20", stride = 1)# 14*14*1024
        self.block3_21 = self.Dense_Block(self.block3_20, name = "conv4_21", stride = 1)# 14*14*1024
        self.block3_22 = self.Dense_Block(self.block3_21, name = "conv4_22", stride = 1)# 14*14*1024
        self.block3_23 = self.Dense_Block(self.block3_22, name = "conv4_23", stride = 1)# 14*14*1024
        self.block3_24 = self.Dense_Block(self.block3_23, name = "conv4_24", stride = 1)# 14*14*1024
        self. blk4 = self.BN_Relu_Conv("conv4_blk", self.block3_24, input_channels = self.block3_24.get_shape().as_list()[-1], 
        								output_channels =self.block3_24.get_shape().as_list()[-1]//2, kernel_size = 1, stride = 1)
        self.pool4 = self.avg_pool(bottom = self. blk4, kernel_size = 2, stride = 2, name = "pool4")
        #stage5

        self.block4_1 = self.Dense_Block(self.pool4, name = "conv5_1", stride = 1)# 8*8*1024
        self.block4_2 = self.Dense_Block(self.block4_1, name = "conv5_2", stride = 1)# 8*8*1024
        self.block4_3 = self.Dense_Block(self.block4_2, name = "conv5_3", stride = 1)# 8*8*1024
        self.block4_4 = self.Dense_Block(self.block4_3, name = "conv5_4", stride = 1)# 8*8*1024
        self.block4_5 = self.Dense_Block(self.block4_4, name = "conv5_5", stride = 1)# 8*8*1024
        self.block4_6 = self.Dense_Block(self.block4_5, name = "conv5_6", stride = 1)# 8*8*1024
        self.block4_7 = self.Dense_Block(self.block4_6, name = "conv5_7", stride = 1)# 8*8*1024
        self.block4_8 = self.Dense_Block(self.block4_7, name = "conv5_8", stride = 1)# 8*8*1024
        self.block4_9 = self.Dense_Block(self.block4_8, name = "conv5_9", stride = 1)# 8*8*1024
        self.block4_10 = self.Dense_Block(self.block4_9, name = "conv5_10", stride = 1)# 8*8*1024
        self.block4_11 = self.Dense_Block(self.block4_10, name = "conv5_11", stride = 1)# 8*8*1024
        self.block4_12 = self.Dense_Block(self.block4_11, name = "conv5_12", stride = 1)# 8*8*1024
        self.block4_13 = self.Dense_Block(self.block4_12, name = "conv5_13", stride = 1)# 8*8*1024
        self.block4_14 = self.Dense_Block(self.block4_13, name = "conv5_14", stride = 1)# 8*8*1024
        self.block4_15 = self.Dense_Block(self.block4_14, name = "conv5_15", stride = 1)# 8*8*1024
        self.block4_16 = self.Dense_Block(self.block4_15, name = "conv5_16", stride = 1)# 8*8*1024
        # upsample layer begins
        self.deconv1_1 = self.deconv_bn_relu(self.block4_16, name = 'deconv1_1',kernel_size = 3, output_channels = 512, 
                        initializer = tf.contrib.layers.variance_scaling_initializer(), stride=2, bn=True, training=self.train_mode)# 14*14
        self.concat1_1 = tf.concat([self.deconv1_1, self.blk4], axis = 3)
        self.res1_0 = self.conv_layer(self.concat1_1, 1, 1024, 512, 1, name = 'res1_0')
        self.res1_1 = self.ResNet_Block(self.res1_0, channel_list = [256,256,512] ,name = "res1_1")# 14*14*512
        self.res1_2 = self.ResNet_Block(self.res1_1, channel_list = [256,256,512] ,name = "res1_2")# 14*14*512

        self.deconv2_1 = self.deconv_bn_relu(self.res1_2, name = 'deconv_2',kernel_size = 3, output_channels = 256, 
                        initializer = tf.contrib.layers.variance_scaling_initializer(), stride=2, bn=True, training=self.train_mode)# 28*28
        self.concat2_1 = tf.concat([self.deconv2_1, self.blk3], axis = 3)
        self.res2_0 = self.conv_layer(self.concat2_1, 1, 512, 256, 1, name = 'res2_0')
        self.res2_1 = self.ResNet_Block(self.res2_0, channel_list = [128,128,256] ,name = "res2_1")# 28*28*256
        self.res2_2 = self.ResNet_Block(self.res2_1, channel_list = [128,128,256] ,name = "res2_2")# 28*28*256


        self.deconv3_2 = self.deconv_bn_relu(self.res2_2, name = 'deconv_3',kernel_size = 3, output_channels = 128, 
                        initializer = tf.contrib.layers.variance_scaling_initializer(), stride=2, bn=True, training=self.train_mode)# 56*56
        self.concat3_1 = tf.concat([self.deconv3_2, self.blk2], axis = 3)
        self.res3_0 = self.conv_layer(self.concat3_1, 1, 256, 128, 1, name = 'res3_0')
        self.res3_1 = self.ResNet_Block(self.res3_0, channel_list = [64,64,128] ,name = "res3_1")# 56*56*128
        self.res3_2 = self.ResNet_Block(self.res3_1, channel_list = [64,64,128] ,name = "res3_2")# 56*56*128



        self.deconv4_3 = self.deconv_bn_relu(self.res3_2, name = 'deconv_4',kernel_size = 3, output_channels = 64, 
                        initializer =tf.contrib.layers.variance_scaling_initializer(), stride=2, bn=True, training=self.train_mode)# 112*112
        self.concat4_1 = tf.concat([self.deconv4_3, self.conv1], axis = 3)
        self.res4_0 = self.conv_layer(self.concat4_1, 1, 128, 64, 1, name = 'res4_0')
        self.res4_1 = self.ResNet_Block(self.res4_0, channel_list = [32,32,64] ,name = "res4_1")# 112*112*64
        self.res4_2 = self.ResNet_Block(self.res4_1, channel_list = [32,32,64] ,name = "res4_2")# 112*112*64

        self.deconv5_4 = self.deconv_bn_relu(self.res4_2, name = 'deconv_5',kernel_size = 3, output_channels = 32, 
                		initializer =tf.contrib.layers.variance_scaling_initializer(), stride=2, bn=True, training=self.train_mode)# 224*224


        self.concat5_1 = tf.concat([self.deconv5_4, self.convd], axis = 3)
        self.res5_0 = self.conv_layer(self.concat5_1, 1, 96, 32, 1, name = 'res5_0')
        self.res5_1 = self.ResNet_Block(self.res5_0, channel_list = [16,16,32] ,name = "res5_1")# 112*112*64
        self.res5_2 = self.ResNet_Block(self.res5_1, channel_list = [16,16,32] ,name = "res5_2")# 112*112*64
        self.res5_3 = self.conv_layer(self.res5_2, 1, 32, 3, 1, name = 'res5_3')
        # self.final_layer = self.conv_layer(bottom = self.deconv_5, kernel_size = 1, in_channels = 64, out_channels = 3, stride = 1, name = 'final_layer')
        # self.final_layer = self.Conv_Bn_Relu(name = 'final_layer', bottom = self.deconv_5, output_channels = 3, kernel_size = 1, stride = 1)
        # self.final_layer = self.conv_bn_relu(bottom = self.deconv_5, name = 'final_layer', kernel_size = 1, output_channels = 3, initializer =tf.contrib.layers.variance_scaling_initializer(), bn = False, training = self.is_training, relu=False)
        # self.pool5 = self.avg_pool(self.block4_3, 7, 1, "pool5")
        #self.fc0 = self.fc_layer(self.pool5, 2048, 1024, "fc0") 
        #self.relu1 = tf.nn.relu(self.fc0)
        #if train_mode is not None:
        #    self.relu1 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu1, self.dropout), lambda: self.relu1)
        #elif self.trainable:
        #    self.relu1 = tf.nn.dropout(self.relu1, self.dropout)
        self.final_layer = self.res5_3
        self.y_soft = tf.nn.softmax(self.final_layer)
        self.logits = tf.reshape(self.final_layer, (-1, 3))
        # print(self.logits)
    # def BN_Relu_DeConv(self, bottom, denseflow, concat_flow, residual_flow, stride):


    # 	return bottom
    def ResNet_Block(self, bottom, channel_list, name):
        input_filter = bottom.get_shape().as_list()[-1]
        block_conv_1 = self.conv_layer(bottom, 1, input_filter, channel_list[0], 1, name + "_branch2a")
        block_norm_1 = tf.layers.batch_normalization(inputs=block_conv_1, axis = 3, momentum=configs['_BATCH_NORM_DECAY'], epsilon=configs['_BATCH_NORM_EPSILON'], center=True, scale=True, training=self.is_training, fused=True)
        block_relu_1 = tf.nn.relu(block_norm_1)

        block_conv_2 = self.conv_layer(block_relu_1, 3, channel_list[0], channel_list[1], 1, name + "_branch2b")
        block_norm_2 = tf.layers.batch_normalization(inputs=block_conv_2, axis = 3, momentum=configs['_BATCH_NORM_DECAY'], epsilon=configs['_BATCH_NORM_EPSILON'], center=True, scale=True, training=self.is_training, fused=True)
        block_relu_2 = tf.nn.relu(block_norm_2)

        block_conv_3 = self.conv_layer(block_relu_2, 1, channel_list[1], channel_list[2], 1, name + "_branch2c")
        block_res = tf.add(bottom, block_conv_3)
        relu = tf.nn.relu(block_res)

        return relu

    # def ResNet_Block(self, bottom, name, stride = 1, output_channels):

    def Dense_Block(self, bottom, name, stride = 1):
        input_channels = bottom.get_shape().as_list()[-1]
        dense_block_1 = self.BN_Relu_Conv(name + '_x1', bottom, input_channels = input_channels, output_channels = K*4, kernel_size = 1, stride = 1)

        dense_block_2 = self.BN_Relu_Conv(name + '_x2', dense_block_1, input_channels = K*4, output_channels = K, kernel_size = 3, stride = 1)
        dense_block = tf.concat([bottom, dense_block_2], axis = 3)
        print('Dense_Block layer {0} -> {1}'.format(bottom.get_shape().as_list(),dense_block.get_shape().as_list()))
        return dense_block

    def BN_Relu_Conv(self, name, bottom, input_channels, output_channels, kernel_size, stride = 1):
       # batch_norm_scale = tf.layers.batch_normalization(name = name + '_bn',inputs=bottom, axis = 3, momentum=configs['_BATCH_NORM_DECAY'],
       #  epsilon=configs['_BATCH_NORM_EPSILON'],center=True, scale=True, training=self.is_training, fused=True)
       batch_norm_scale = self.batch_norm_layer(name, bottom,phase_train = self.train_mode)
       relu = tf.nn.relu(batch_norm_scale)
       conv = self.conv_layer(bottom = relu, kernel_size = kernel_size, in_channels = input_channels, 
       							out_channels = output_channels, stride = stride, name = name)
       return conv
       
    def Conv_Bn_Relu(self, name, bottom, output_channels, kernel_size, stride = 1):
    	input_channels = bottom.get_shape().as_list()[-1]
    	conv = self.conv_layer(bottom = bottom, kernel_size = kernel_size, in_channels = input_channels, 
    							out_channels = output_channels, stride = stride, name = name)
    	batch_norm_scale = self.batch_norm_layer(name = name, bottom = conv, phase_train = self.train_mode)
    	# batch_norm_scale = tf.layers.batch_normalization(inputs=conv, axis = 3, momentum=configs['_BATCH_NORM_DECAY'], epsilon=configs['_BATCH_NORM_EPSILON'],
    	# 						center=True, scale=True, training=self.train_mode, fused=True)
    	relu = tf.nn.relu(batch_norm_scale)
    	
    	return conv


    def avg_pool(self,bottom, kernel_size = 2, stride = 2, name = "avg"):
    	avg_pool = tf.nn.avg_pool(bottom, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding='SAME', name=name)
    	print('avg_pool layer {0} -> {1}'.format(bottom.get_shape().as_list(),avg_pool.get_shape().as_list()))
    	return avg_pool

    def max_pool(self,bottom, kernel_size = 3, stride = 2, name = "max"):
    	# paddings = [[0,0],[1,1],[1,1],[0,0]]
    	# padded = tf.pad(bottom,paddings,"CONSTANT")
    	max_pool = tf.nn.max_pool(bottom, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding='SAME', name=name)
    	print('max_pool layer {0} -> {1}'.format(bottom.get_shape().as_list(),max_pool.get_shape().as_list()))
    	return max_pool

    def conv_layer(self, bottom, kernel_size, in_channels, out_channels, stride, name):
    	with tf.variable_scope(name):
    		filt, conv_biases = self.get_conv_var(kernel_size, in_channels, out_channels, name)
    		conv = tf.nn.conv2d(bottom, filt, [1,stride,stride,1], padding='SAME')
    		bias = tf.nn.bias_add(conv, conv_biases)

    		tf.summary.histogram('weight', filt)
    		tf.summary.histogram('bias', conv_biases)

    		return bias

    def conv_bn_relu(self, bottom,name, kernel_size, output_channels, initializer,stride=1, bn=False,training=False,relu=True):
        input_channels = bottom.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            kernel = self.variable('weights', [kernel_size, kernel_size, input_channels, output_channels], initializer, regularizer=tf.contrib.layers.l2_regularizer(0.0005))
            conv = tf.nn.conv2d(bottom, kernel, [1, stride, stride, 1], padding='SAME')
            biases = self.variable('biases', [output_channels], tf.constant_initializer(0.0))
            conv_layer = tf.nn.bias_add(conv, biases)
            if bn:
                conv_layer = self.batch_norm_layer(name, bottom = conv_layer, phase_train = self.train_mode)
                # self.batch_norm_layer(conv_layer,scope,training)
            if relu:
                conv_layer = tf.nn.relu(conv_layer, name=scope.name)
        print('Conv layer {0} -> {1}'.format(bottom.get_shape().as_list(),conv_layer.get_shape().as_list()))
        return conv_layer

    def batch_norm_layer(self, name, bottom, phase_train):
    	n_out = bottom.get_shape().as_list()[-1]
    	mean,var,gamma,beta = self.get_batchnorm_var(n_out, name + '_bn')
    	batch_mean, batch_var = tf.nn.moments(bottom, [0], name='moments')
    	ema = tf.train.ExponentialMovingAverage(decay=0.5)
    	def mean_var_with_update():
    		ema_apply_op = ema.apply([batch_mean, batch_var])
    		with tf.control_dependencies([ema_apply_op]):
    			return tf.identity(batch_mean), tf.identity(batch_var)
    	# tf.cond(phase_train, print('train_phase.............................'))
    	mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
    	normed = tf.nn.batch_normalization(bottom, mean, var, beta, gamma, configs['_BATCH_NORM_EPSILON'])

    	return normed



    def deconv_bn_relu(self, bottom, name, kernel_size, output_channels, initializer, stride = 2, bn=False, training=False, relu=True):
    	deconv_layer = self.deconv_layer(bottom, name, output_channels, kernel_size, stride, regularizer=None)
    	if bn:
    		deconv_layer = self.batch_norm_layer(name, bottom = deconv_layer, phase_train = self.train_mode)
    	if relu:
    		deconv_layer = tf.nn.relu(deconv_layer, name=name)

    	print('Deconv layer {0} -> {1}'.format(bottom.get_shape().as_list(), deconv_layer.get_shape().as_list()))

    	return deconv_layer



    def variable(self, name, shape, initializer,regularizer=None):
        with tf.device('/cpu:0'):
            return tf.get_variable(name, shape, initializer=initializer, regularizer=regularizer, trainable=True)

    def deconv_layer(self, bottom, name, output_channels, kernel_size, stride, regularizer=None):
    	input_shape = bottom.get_shape().as_list()
    	output_shape = [input_shape[0], input_shape[1]*stride, input_shape[2]*stride, output_channels]
    	kernel_shape = [kernel_size, kernel_size, output_channels, input_shape[-1]]

    	initial_weights = tf.truncated_normal(shape = kernel_shape, mean = 0, stddev = 1 / math.sqrt(float(kernel_size * kernel_size)))
    	weights = self.get_var(initial_value = initial_weights, name =  name, idx = 'weights',  var_name = name + "_weights")
    	initial_biases = tf.truncated_normal([output_channels], 0.0, 1.0)
    	biases = self.get_var(initial_value = initial_biases, name =  name, idx = 'biases',  var_name = name + "_biases")
    	deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape, [1, stride, stride, 1], padding='SAME')
    	deconv_layer = tf.nn.bias_add(deconv, biases)

    	return deconv_layer






    def fc_layer(self, bottom, in_size, out_size, name):
    	with tf.variable_scope(name):
    		weights, biases = self.get_fc_var(in_size, out_size, name)

    		x = tf.reshape(bottom, [-1, in_size])
    		fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

    		tf.summary.histogram('weight', weights)
    		tf.summary.histogram('bias', biases)

    		return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, stddev = 1 / math.sqrt(float(filter_size * filter_size)))
        filters = self.get_var(initial_value = initial_value, name =  name, idx = 'weights',  var_name = "_filters")
        
        initial_value = tf.truncated_normal([out_channels], 0.0, 1.0)
        biases = self.get_var(initial_value = initial_value, name = name, idx = 'biases', var_name = "_biases")
        
        return filters, biases

    def get_batchnorm_var(self, n_out, name):
    	"""batch normal parameters:
    	1. mean(batch_var) 2. variance(batch_mean) 3. scale(gamma) 4. offset(beta)
    	"""
    	init_beta = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
    	beta = self.get_var(initial_value = init_beta, name =  name, idx = 'offset',  var_name = name + "_offset")

    	init_gamma = tf.constant(1.0, shape=[n_out],dtype=tf.float32)
    	gamma = self.get_var(initial_value = init_gamma, name =  name, idx = 'scale',  var_name = name + "_scale")

    	init_variance = tf.constant(1.0, shape=[n_out], dtype=tf.float32)
    	variance = self.get_var(initial_value = init_variance, name =  name, idx = 'variance',  var_name = name + "_variance")

    	init_mean = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
    	mean = self.get_var(initial_value = init_mean, name =  name, idx = 'mean',  var_name = name + "_mean")

    	return mean, variance, gamma, beta


    def get_fc_var(self, in_size, out_size, name):
    	"""
    	in_size : number of input feature size
    	out_size : number of output feature size
    	name : block_layer name
    	"""
    	initial_value = tf.truncated_normal([in_size, out_size], 0.0, stddev = 1 / math.sqrt(float(in_size)))
    	weights = self.get_var(initial_value, name, 0, name + "_weights")

    	initial_value = tf.truncated_normal([out_size], 0.0, 1.0)
    	biases = self.get_var(initial_value, name, 1, name + "_biases")

    	return weights, biases


    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict.keys() and idx in self.data_dict[name]:
            value = self.data_dict[name][idx]
        else:
            value = initial_value
            
        if self.trainable:
            var = tf.get_variable(name = var_name, initializer=value, trainable=True)
            # tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)
            
        self.var_dict[(name, idx)] = var
        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()
        
        return var


    def save_npy(self, sess, npy_path="./Resnet-save.npy"):
    	"""
    	Save this model into a npy file
    	"""
    	assert isinstance(sess, tf.Session)

    	data_dict = {}

    	for (name, idx), var in list(self.var_dict.items()):
    		var_out = sess.run(var)
    		if name not in data_dict:
    			data_dict[name] = {}
    		data_dict[name][idx] = var_out

    	np.save(npy_path, data_dict)
    	print(("file saved", npy_path))
    	return npy_path

    def get_var_count(self):
    	count = 0
    	for v in list(self.var_dict.values()):
    		count += reduce(lambda x, y: x * y, v.get_shape().as_list())
    	return count

