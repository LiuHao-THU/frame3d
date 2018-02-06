"""
Time: 2017.12.27
Author: LiuHao
Institution:THU
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

"""
This code is the first edition of the dense net for segmentation 
The encode part use the pretrained network from the "https://github.com/shicai/DenseNet-Caffe"
we use transpose convolution and deconv operation 
"""


import math
import numpy as np
import tensorflow as tf
from functools import reduce
from tensorflow.python.ops import random_ops
import six
from tensorflow.python.training.moving_averages import assign_moving_average
from configs import configs
K = 32
middle_layer = K * 4

class ResNet:
	# some properties
    """
    Initialize function
    """
    def __init__(self, npy_path=None, trainable=True, open_tensorboard=False, dropout=0.8):
        if npy_path is not None:
            self.data_dict = np.load(npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.open_tensorboard = open_tensorboard
        self.dropout = dropout

    def set_is_training(self, isTrain):
    	self.is_training = isTrain

    def build(self, rgb, label_num, train_mode=None, last_layer_type = "softmax"):
        """
        load variable from npy to build the Resnet or Generate a new one
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """
        self.train_mode = train_mode
        rgb = (rgb -rgb.mean())/(rgb.std())
        print(bgr.get_shape().as_list())
        #convolution part 
        self.conv1a = self.Conv_Bn_Relu(name = 'conv1a', bottom = bgr, output_channels = 32, kernel_size = [3,3,3], stride = 1)
        self.conv1b = self.Conv_Bn_Relu(name = 'conv1b', bottom = self.conv1a, output_channels = 32, kernel_size = [1,3,3], stride = 1)
        self.conv1c = self.Conv_Bn_Relu(name = 'conv1c', bottom = self.conv1b, output_channels = 64, kernel_size = [3,3,3], stride = 2, relu = False, bn = False)
        self.VoxRes2 = self.VoxRes_3d(name = 'VoxRes2', bottom = self.conv1c, channel_list = [64,64])
        self.VoxRes3 = self.VoxRes_3d(name = 'VoxRes3', bottom = self.VoxRes2, channel_list = [64,64])
        self.conv4 = self.BN_Relu_Conv(name = 'conv4', bottom = self.VoxRes3, output_channels = 64, kernel_size = [3,3,3], stride = 2)
        self.VoxRes5 = self.VoxRes_3d(name = 'VoxRes5', bottom = self.conv4, channel_list = [64,64])
        self.VoxRes6 = self.VoxRes_3d(name = 'VoxRes6', bottom = self.VoxRes5, channel_list = [64,64])
        self.conv7 = self.BN_Relu_Conv(name = 'conv7', bottom = self.VoxRes6, output_channels = 64, kernel_size = [3,3,3], stride = 2)
        self.VoxRes8 = self.VoxRes_3d(name = 'VoxRes8', bottom = self.conv7, channel_list = [64,64])
        self.VoxRes9 = self.VoxRes_3d(name = 'VoxRes9', bottom = self.VoxRes8, channel_list = [64,64])

        #deconvolution part
        self.deconv1_1 = self.Deconv_Bn_Relu(bottom = self.VoxRes9, name = 'deconv1_1', kernel_size = [3,3,3], 
                                                                            64, stride = 2, bn=False, training=False, relu=True)
        self.conv1_1 = self.VoxRes_3d(name = 'conv1_1', bottom = self.deconv1_1, channel_list = [64,64])
        self.concat1 = tf.concat(self.VoxRes6,self.conv1_1)

        self.deconv2_1 = self.Deconv_Bn_Relu(bottom = self.concat1, name = 'deconv2_1', kernel_size = [3,3,3], 
                                                                            64, stride = 2, bn=False, training=False, relu=True)
        self.conv2_1 = self.VoxRes_3d(name = 'conv2_1', bottom = self.deconv2_1, channel_list = [64,64])
        self.concat2 = tf.concat(self.VoxRes6,self.conv1a)

        self.deconv3_1 = self.Deconv_Bn_Relu(bottom = self.concat2, name = 'deconv3_1', kernel_size = [3,3,3], 
                                                                            64, stride = 2, bn=False, training=False, relu=True)
        self.conv3_1 = self.VoxRes_3d(name = 'conv3_1', bottom = self.deconv1_1, channel_list = [64,64])
        self.concat3 = tf.concat(self.VoxRes6,self.conv1a)

        self.deconv4_1 = self.Deconv_Bn_Relu(bottom = self.concat3, name = 'deconv4_1', kernel_size = [3,3,3], 
                                                                            32, stride = 2, bn=False, training=False, relu=True)
        self.conv4_1 = self.VoxRes_3d(name = 'conv4_1', bottom = self.deconv1_1, channel_list = [64,64])
        self.concat4 = tf.concat(self.VoxRes6,self.conv1a)





        self.y_soft = tf.nn.softmax(self.final_layer)
        output_shape = self.final_layer.get_shape().as_list()
        self.logits = tf.reshape(self.final_layer,[output_shape[0],-1, 3])
        # self.logits = tf.reshape(self.final_layer, (-1, 3))
        self.pred = tf.argmax(self.y_soft, axis = 3)



    def VoxRes_3d(self, name, bottom, channel_list):
        input_filter = bottom.get_shape().as_list()[-1]
        block_conv_1 = self.BN_Relu_Conv(name = name + "_branch2a", bottom = bottom, output_channels = channel_list[0], kernel_size = [1,3,3], stride = 1)
        block_conv_2 = self.BN_Relu_Conv(name = name + "_branch2b", bottom = bottom, output_channels = channel_list[1], kernel_size = [3,3,3], stride = 1)
        block_res = tf.add(bottom, block_conv_2)
        relu = tf.nn.relu(block_res)
        return relu


    def Conv_Bn_Relu(self, name, bottom, output_channels, kernel_size, stride = 1, relu = True, bn = True):

        input_channels = bottom.get_shape().as_list()[-1]
        conv_layer = self.Conv_Layer3d(bottom = bottom, kernel_size = kernel_size, in_channels = input_channels, 
            out_channels = output_channels, stride = stride, regularizer=tf.contrib.layers.l2_regularizer(0.0005) ,name = name)

        if bn == True:
            batch_norm_scale = self.Batch_Norm_Layer3d(name = name, bottom = conv_layer, phase_train = self.train_mode)
        else:
            batch_norm_scale = conv_layer

        if relu == True:
            relu_layer = tf.nn.relu(batch_norm_scale)
        else:
            relu_layer = batch_norm_scale

        return relu_layer
        
    def avg_pool(self,bottom, kernel_size = 2, stride = 2, name = "avg"):
    	avg_pool = tf.nn.avg_pool(bottom, ksize=[1, kernel_size[0], kernel_size[1], kernel_size[2], 1], strides=[1, stride, stride, stride, 1], padding='SAME', name=name)
    	print('avg_pool layer {0} -> {1}'.format(bottom.get_shape().as_list(),avg_pool.get_shape().as_list()))
    	return avg_pool

    def max_pool(self,bottom, kernel_size = [2,2,2], stride = 2, name = "max"):
    	max_pool = tf.nn.max_pool(bottom, ksize=[1, kernel_size[0], kernel_size[1], kernel_size[2], 1], strides=[1, stride, stride, stride, 1], padding='SAME', name=name)
    	print('max_pool layer {0} -> {1}'.format(bottom.get_shape().as_list(),max_pool.get_shape().as_list()))
    	return max_pool

    def BN_Relu_Conv(self, name, bottom, output_channels, kernel_size = [3,3,3], stride = 1):
        input_channels = bottom.get_shape().as_list()
        batch_norm_scale = self.Batch_Norm_Layer3d(name, bottom, phase_train = self.train_mode)
        relu = tf.nn.relu(batch_norm_scale)
        conv = self.Conv_Layer3d(bottom = relu, kernel_size = kernel_size, in_channels = input_channels, 
                                out_channels = output_channels, stride = stride, name = name)
        return conv

    def Conv_Layer3d(self, bottom, kernel_size = [3,3,3], in_channels, out_channels, stride, name, regularizer = None):
        with tf.variable_scope(name):
            filt, conv_biases = self.Get_Conv_Var3d(kernel_size, in_channels, out_channels, name, regularizer = regularizer)
            conv = tf.nn.conv3d(bottom, filt, [1,stride,stride,stride,1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)

            tf.summary.histogram('weight', filt)
            tf.summary.histogram('bias', conv_biases)

            return bias


    def Bn_Relu(self, bottom, phase_train, name):
        batch_norm_scale = self.Batch_Norm_Layer3d(name, bottom, phase_train = self.train_mode)
        return relu = tf.nn.relu(batch_norm_scale)




    def Deconv_layer3d(self, bottom, name, output_channels, kernel_size, stride, regularizer=None):
        input_shape = bottom.get_shape().as_list()
        output_shape = [input_shape[0], input_shape[1]*stride, input_shape[2]*stride, input_shape[3]*stride, output_channels]
        kernel_shape = [kernel_size[0], kernel_size[1], kernel_size[2], output_channels, input_shape[-1]]

        initial_weights = tf.truncated_normal(shape = kernel_shape, mean = 0, stddev = 1 / math.sqrt(float(kernel_size * kernel_size)))
        weights = self.get_var(initial_value = initial_weights, name =  name, idx = 'weights',  var_name = name + "_weights")
        initial_biases = tf.truncated_normal([output_channels], 0.0, 1.0)
        biases = self.get_var(initial_value = initial_biases, name =  name, idx = 'biases',  var_name = name + "_biases")
        deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape, [1, stride, stride, stride, 1], padding='SAME')
        deconv_layer = tf.nn.bias_add(deconv, biases)

        return deconv_layer


    def Batch_Norm_Layer3d(self, name, bottom, phase_train, decay=0.5):
        """
        glabal batch norm with input [batch_size height width channel]
        """
        n_out = bottom.get_shape().as_list()[-1]
        #restore the stored moving_mean moving_variance, beta, gamma if use pretrained model
        moving_mean,moving_variance,gamma,beta = self.Get_Batchnorm_Var(n_out, name + '_bn')

        def mean_var_with_update():
            #if train model updata the moving mean and moving variance
            mean, variance = tf.nn.moments(bottom, [0,1,2,3], name='moments')
            with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay),
                                          assign_moving_average(moving_variance, variance, decay)]):
                return tf.identity(mean), tf.identity(variance)
        # if test eval model use the moving restored moving mean and moving_variance.
        mean, variance = tf.cond(phase_train, mean_var_with_update, lambda: (moving_mean, moving_variance))
        return tf.nn.batch_normalization(bottom, mean, variance, beta, gamma, configs['_BATCH_NORM_EPSILON'])



    def Deconv_Bn_Relu(self, bottom, name, kernel_size, output_channels, stride = 2, bn=False, training=False, relu=True):
        deconv_layer = self.Deconv_layer3d(bottom, name, output_channels, kernel_size, stride, regularizer=None)
        if bn:
            deconv_layer = self.Batch_Norm_Layer3d(name, bottom = deconv_layer, phase_train = self.train_mode)
        if relu:
            deconv_layer = tf.nn.relu(deconv_layer, name=name)

        print('Deconv layer {0} -> {1}'.format(bottom.get_shape().as_list(), deconv_layer.get_shape().as_list()))

        return deconv_layer

    def upsample_layer(self, bottom, name, output_channels, kernel_size, stride, regularizer=None):
    	kernel_size = 2*factor - factor%2   #if factor = 2 kernel_size = 3
    	return bottom


	def upsample_filt(self, size):
	    """
	    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
	    """
	    factor = (size + 1) // 2
	    if size % 2 == 1:
	        center = factor - 1
	    else:
	        center = factor - 0.5
	    og = np.ogrid[:size, :size]
	    return (1 - abs(og[0] - center) / factor) * \
	           (1 - abs(og[1] - center) / factor)


    def bilinear_interpolation(self, bottom):


    	return bottom


    def Get_Conv_Var3d(self, filter_size = [3,3,3], in_channels, out_channels, name, regularizer = None):
        """get the conv weights and baises in pretrained model if not initialize with ramdom numbers.
        """
        trunc_stddev = self.Variance_Scaling_Initializer3d(shape = [filter_size[0], filter_size[1], filter_size[2], in_channels, out_channels],
                                         mode = 'FAN_AVG', uniform = False, factor = 2)
        initial_value = tf.truncated_normal([filter_size[0], filter_size[1], filter_size[2], in_channels, out_channels], 0.0, stddev = trunc_stddev)
        filters = self.get_var(initial_value = initial_value, name =  name, idx = 'weights',  var_name = "_filters",regularizer = regularizer)
        
        initial_value = tf.truncated_normal([out_channels], 0.0, 1.0)
        biases = self.get_var(initial_value = initial_value, name = name, idx = 'biases', var_name = "_biases")
        
        return filters, biases

    def Variance_Scaling_Initializer3d(self, shape, mode = 'FAN_AVG', uniform = False, factor = 2):
        """
        initialization
        """
        fin_in = shape[-2]*shape[0]*shape[1]*shape[2]
        fin_out = shape[-1]*shape[0]*shape[1]*shape[2]
        if mode == "FAN_IN":
            n = fin_in
        elif mode == "FAN_OUT":
            n = fin_out
        elif mode == "FAN_AVG":
            n = (fin_in+ fin_out)/2
        if uniform ==False:
            
            return math.sqrt(1.3*factor/n)
        else:
            return math.sqrt(3.0 * factor / n)



    def Get_Batchnorm_Var(self, n_out, name):
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


    def get_var(self, initial_value, name, idx, var_name, regularizer = None, trainable=True):
        """
        detect if the pretrained model has the variable if not creat one as expected
        """
        if self.data_dict is not None and name in self.data_dict.keys() and idx in self.data_dict[name]:
            value = self.data_dict[name][idx]
        else:
            value = initial_value
            
        if self.trainable:
            var = tf.get_variable(name = var_name, initializer=value, trainable=trainable, regularizer = regularizer)
            # tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)
            
        self.var_dict[(name, idx)] = var
        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()
        
        return var


    def save_npy(self, sess, npy_path="./model_saved.npy"):
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

    def mean_iou(self, logits, labels):
        #logits imported from the last layer of the net[]
        #
        return logits
