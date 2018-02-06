"""
A Trainable ResNet Class is defined in this file
Author: Kaihua Tang
"""
import math
import numpy as np
import tensorflow as tf
from functools import reduce

VGG_MEAN = [104.7546, 124.328, 167.1754]
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
he_initializer = tf.contrib.layers.variance_scaling_initializer()
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
    def set_is_training(self, isTrain):
    	self.is_training = isTrain

    def build(self, rgb, label_num, train_mode=None, last_layer_type = "softmax"):
        """
        load variable from npy to build the Resnet or Generate a new one
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        print(bgr.get_shape().as_list())
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        #stage2
        self.conv1 = self.conv_layer(bgr, 7, 3, 64, 2, "conv1")# 112*112
        self.pool1 = self.max_pool(self.conv1, 3, 2, "pool1")# 56*56* 64
        self.block1_1 = self.res_block_3_layers(self.pool1, [64, 64, 256], "res2a", True)# 56*56*256
        self.block1_2 = self.res_block_3_layers(self.block1_1, [64, 64, 256], "res2b")# 56*56*256
        self.block1_3 = self.res_block_3_layers(self.block1_2, [64, 64, 256], "res2c")# 56*56*256

        #stage3
        self.block2_1 = self.res_block_3_layers(self.block1_3, [128, 128, 512], "res3a", True, stride = 2)# 28*28*512
        self.block2_2 = self.res_block_3_layers(self.block2_1, [128, 128, 512], "res3b")# 28*28*512
        self.block2_3 = self.res_block_3_layers(self.block2_2, [128, 128, 512], "res3c")# 28*28*512
        self.block2_4 = self.res_block_3_layers(self.block2_3, [128, 128, 512], "res3d")# 28*28*512
        #stage4
        self.block3_0 = self.res_block_3_layers(self.block2_4, [256, 256, 1024], "res4a", True, stride = 2)# 14*14*1024
        self.block3_1 = self.res_block_3_layers(self.block3_0, [256, 256, 1024], "res4b1")# 14*14*1024
        self.block3_2 = self.res_block_3_layers(self.block3_1, [256, 256, 1024], "res4b2")# 14*14*1024
        self.block3_3 = self.res_block_3_layers(self.block3_2, [256, 256, 1024], "res4b3")# 14*14*1024
        self.block3_4 = self.res_block_3_layers(self.block3_3, [256, 256, 1024], "res4b4")# 14*14*1024
        self.block3_5 = self.res_block_3_layers(self.block3_4, [256, 256, 1024], "res4b5")# 14*14*1024
        self.block3_6 = self.res_block_3_layers(self.block3_5, [256, 256, 1024], "res4b6")# 14*14*1024
        self.block3_7 = self.res_block_3_layers(self.block3_6, [256, 256, 1024], "res4b7")# 14*14*1024
        self.block3_8 = self.res_block_3_layers(self.block3_7, [256, 256, 1024], "res4b8")# 14*14*1024
        self.block3_9 = self.res_block_3_layers(self.block3_8, [256, 256, 1024], "res4b9")# 14*14*1024
        self.block3_10 = self.res_block_3_layers(self.block3_9, [256, 256, 1024], "res4b10")# 14*14*1024
        self.block3_11 = self.res_block_3_layers(self.block3_10, [256, 256, 1024], "res4b11")# 14*14*1024
        self.block3_12 = self.res_block_3_layers(self.block3_11, [256, 256, 1024], "res4b12")# 14*14*1024
        self.block3_13 = self.res_block_3_layers(self.block3_12, [256, 256, 1024], "res4b13")# 14*14*1024
        self.block3_14 = self.res_block_3_layers(self.block3_13, [256, 256, 1024], "res4b14")# 14*14*1024
        self.block3_15 = self.res_block_3_layers(self.block3_14, [256, 256, 1024], "res4b15")# 14*14*1024
        self.block3_16 = self.res_block_3_layers(self.block3_15, [256, 256, 1024], "res4b16")# 14*14*1024
        self.block3_17 = self.res_block_3_layers(self.block3_16, [256, 256, 1024], "res4b17")# 14*14*1024
        self.block3_18 = self.res_block_3_layers(self.block3_17, [256, 256, 1024], "res4b18")# 14*14*1024
        self.block3_19 = self.res_block_3_layers(self.block3_18, [256, 256, 1024], "res4b19")# 14*14*1024
        self.block3_20 = self.res_block_3_layers(self.block3_19, [256, 256, 1024], "res4b20")# 14*14*1024
        self.block3_21 = self.res_block_3_layers(self.block3_20, [256, 256, 1024], "res4b21")# 14*14*1024
        self.block3_22 = self.res_block_3_layers(self.block3_21, [256, 256, 1024], "res4b22")# 14*14*1024
        #stage5
        self.block4_1 = self.res_block_3_layers(self.pool4, [512, 512, 2048], "res5a", True, stride = 2)# 7*7
        self.block4_2 = self.res_block_3_layers(self.block4_1, [512, 512, 2048], "res5b")# 7*7
        self.block4_3 = self.res_block_3_layers(self.block4_2, [512, 512, 2048], "res5c")# 7*7


        # upsample layer begins
        self.deconv_1 = self.deconv_bn_relu(self.block4_3, name = 'deconv_1',kernel_size = 3, output_channels = 1024, 
                        initializer = tf.contrib.layers.variance_scaling_initializer(), stride=2, bn=True, training=self.is_training)# 14*14
        self.deconv_2 = self.deconv_bn_relu(self.deconv_1, name = 'deconv_2',kernel_size = 3, output_channels = 512, 
                        initializer = tf.contrib.layers.variance_scaling_initializer(), stride=2, bn=True, training=self.is_training)# 28*28
        self.deconv_3 = self.deconv_bn_relu(self.deconv_2, name = 'deconv_3',kernel_size = 3, output_channels = 256, 
                        initializer = tf.contrib.layers.variance_scaling_initializer(), stride=2, bn=True, training=self.is_training)# 56*56
        self.deconv_4 = self.deconv_bn_relu(self.deconv_3, name = 'deconv_4',kernel_size = 3, output_channels = 128, 
                        initializer =tf.contrib.layers.variance_scaling_initializer(), stride=2, bn=True, training=self.is_training)# 112*112
        self.deconv_5 = self.deconv_bn_relu(self.deconv_4, name = 'deconv_5',kernel_size = 3, output_channels = 64, 
                initializer =tf.contrib.layers.variance_scaling_initializer(), stride=2, bn=True, training=self.is_training)# 224*224
        
        # self.final_layer = self.conv_layer(bottom = self.deconv_5, kernal_size = 1, in_channels = 64, out_channels = 3, stride = 1, name = 'final_layer')
        self.final_layer = self.conv_bn_relu(bottom = self.deconv_5, name = 'final_layer', kernel_size = 1, output_channels = 3, initializer =tf.contrib.layers.variance_scaling_initializer(), bn = False, training = self.is_training, relu=False)
        # self.pool5 = self.avg_pool(self.block4_3, 7, 1, "pool5")
        #self.fc0 = self.fc_layer(self.pool5, 2048, 1024, "fc0") 
        #self.relu1 = tf.nn.relu(self.fc0)
        #if train_mode is not None:
        #    self.relu1 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu1, self.dropout), lambda: self.relu1)
        #elif self.trainable:
        #    self.relu1 = tf.nn.dropout(self.relu1, self.dropout)

        self.y_soft = tf.nn.softmax(self.final_layer)
        self.logits = tf.reshape(self.final_layer, (-1, 3))
        print(self.logits)

        # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=logits, name=None)


        # self.loss = tf.reduce_mean(cross_entropy, name = 'xcross_entropy')
        # if(last_layer_type == "sigmoid"):
	       #  self.prob = tf.nn.sigmoid(self.fc1, name="prob")
        # elif(last_layer_type == "softmax"):
        # 	self.prob = tf.nn.softmax(self.fc1, name="prob")

        self.data_dict = None



    def res_block_3_layers(self, bottom, channel_list, name, change_dimension = False, stride = 1):
        if (change_dimension):
            block_conv_input = self.conv_layer(bottom = bottom, kernal_size = 1, in_channels = bottom.get_shape().as_list()[-1], 
                                               out_channels = channel_list[2], stride = 1, name = name + "_branch1")
        else:
            block_conv_input = bottom
        input_filter = bottom.get_shape().as_list()[-1]
        block_conv_1 = self.conv_layer(bottom, 1, input_filter, channel_list[0], stride = stride, name + "_branch2a")
        block_norm_1 = tf.layers.batch_normalization(inputs=block_conv_1, axis = 3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.is_training, fused=True)
        block_relu_1 = tf.nn.relu(block_norm_1)
        
        block_conv_2 = self.conv_layer(block_relu_1, 3, channel_list[0], channel_list[1], 1, name + "_branch2b")
        block_norm_2 = tf.layers.batch_normalization(inputs=block_conv_2, axis = 3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=self.is_training, fused=True)
        block_relu_2 = tf.nn.relu(block_norm_2)
        
        block_conv_3 = self.conv_layer(block_relu_2, 1, channel_list[1], channel_list[2], 1, name + "_branch2c")
        block_res = tf.add(block_conv_input, block_conv_3)
        relu = tf.nn.relu(block_res)
        
        return relu


    def avg_pool(self, bottom, kernal_size = 2, stride = 2, name = "avg"):
    	return tf.nn.avg_pool(bottom, ksize=[1, kernal_size, kernal_size, 1], strides=[1, stride, stride, 1], padding='VALID', name=name)

    def max_pool(self, bottom, kernal_size = 2, stride = 2, name = "max"):
    	return tf.nn.max_pool(bottom, ksize=[1, kernal_size, kernal_size, 1], strides=[1, stride, stride, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, kernal_size, in_channels, out_channels, stride, name):
    	with tf.variable_scope(name):
    		filt, conv_biases = self.get_conv_var(kernal_size, in_channels, out_channels, name)
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
                conv_layer = self.batch_norm_layer(conv_layer,scope,training)
            if relu:
                conv_layer = tf.nn.relu(conv_layer, name=scope.name)
        print('Conv layer {0} -> {1}'.format(bottom.get_shape().as_list(),conv_layer.get_shape().as_list()))
        return conv_layer

    def batch_norm_layer(self, input_tensor,scope,training):
        return tf.contrib.layers.batch_norm(input_tensor,scope=scope,is_training=training,decay=0.99)

    def deconv_bn_relu(self, bottom, name, kernel_size, output_channels, initializer, stride = 1, bn=False, training=False, relu=True):
        input_shape = bottom.get_shape().as_list()
        input_channels = input_shape[-1]
        output_shape = [input_shape[0], input_shape[1]*stride, input_shape[2]*stride, output_channels]
        with tf.variable_scope(name) as scope:
            kernel = self.variable('weights', [kernel_size, kernel_size, output_channels, input_channels], initializer, regularizer=tf.contrib.layers.l2_regularizer(0.0005))
            deconv = tf.nn.conv2d_transpose(bottom, kernel, output_shape, [1, stride, stride, 1], padding='SAME')
            biases = self.variable('biases', [output_channels], tf.constant_initializer(0.0))
            deconv_layer = tf.nn.bias_add(deconv, biases)
            if bn:
                deconv_layer = self.batch_norm_layer(deconv_layer,scope,training)
            if relu:
                deconv_layer = tf.nn.relu(deconv_layer, name=scope.name)
        print('Deconv layer {0} -> {1}'.format(bottom.get_shape().as_list(),deconv_layer.get_shape().as_list()))
        return deconv_layer



    def variable(self, name, shape, initializer,regularizer=None):
        with tf.device('/cpu:0'):
            return tf.get_variable(name, shape, initializer=initializer, regularizer=regularizer, trainable=True)


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
        if self.data_dict is not None and idx in self.data_dict[name]:
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

    # def batch_norm_scale(self, bottom, use_bias = True,):
    #     bottom_shape = bottom.get_shape()
    #     params_shape = bottom_shape[-1:]

    #     if use_bias:
    #         bias = _get_variable('bias', params_shape,
    #                              initializer=tf.zeros_initializer)
    #         return x + bias


    #     axis = list(range(len(x_shape) - 1))

    #     beta = _get_variable('beta',
    #                          params_shape,
    #                          initializer=tf.zeros_initializer)
    #     gamma = _get_variable('gamma',
    #                           params_shape,
    #                           initializer=tf.ones_initializer)

    #     moving_mean = _get_variable('moving_mean',
    #                                 params_shape,
    #                                 initializer=tf.zeros_initializer,
    #                                 trainable=False)
    #     moving_variance = _get_variable('moving_variance',
    #                                     params_shape,
    #                                     initializer=tf.ones_initializer,
    #                                     trainable=False)

    #     # These ops will only be preformed when training.
    #     mean, variance = tf.nn.moments(x, axis)
    #     update_moving_mean = moving_averages.assign_moving_average(moving_mean,
    #                                                                mean, BN_DECAY)
    #     update_moving_variance = moving_averages.assign_moving_average(
    #         moving_variance, variance, BN_DECAY)
    #     tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    #     tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    #     mean, variance = control_flow_ops.cond(
    #         c['is_training'], lambda: (mean, variance),
    #         lambda: (moving_mean, moving_variance))

    #     x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
    #     #x.set_shape(inputs.get_shape()) ??

    #     return x