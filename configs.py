"""
this .py file contains all the parameters
"""
import os
configs = {}

#****************************************read data parameters**************************************
configs['max_angle'] = 20
configs['root_dir'] = 'test_data/data'
configs['save_dir'] = 'test_data/saved_data'
configs['image_size'] = 224
configs['per'] = 0.9            #percentage splited from the raw data
configs['saved_npy'] = True
configs['imgs_train'] = 'imgs_train.npy'
configs['imgs_label'] = 'imgs_label.npy'
configs['imgs_train_test'] = 'imgs_train_test.npy'
configs['imgs_label_test'] = 'imgs_label_test.npy'
# configs['model_path'] = "./model/resnet/resnet101.npy"
configs['model_path'] = './model/densenet/1.npy'
#**************************************argumentation parameters************************************
configs['raw_images'] = True
configs['horizontal_flip_num'] = True
configs['vertical_flip_num'] = True
configs['random_rotate_num'] = 0
configs['random_crop_num'] = 0
configs['center_crop_num'] = 0
configs['slide_crop_num'] = 0
configs['slide_crop_old_num'] = 0
#*************************************train parameters**********************************************
configs['image_size'] = 224
# configs['channel'] = 3
configs['channel'] = 3
configs["batch_size"] = 32
configs['epoch'] = 200
configs['final_layer_type'] = "softmax_sparse"
configs['learning_rate_orig'] = 1e-2
configs['checkpoint_dir'] = 'model/'
configs['num_classes'] = 3
configs['VGG_MEAN'] = [104.7546, 124.328, 167.1754]
configs['_BATCH_NORM_DECAY'] = 0.997
configs['_BATCH_NORM_EPSILON'] = 1e-5
configs['save_frequency'] = 100
#************************************device parameters**********************************************
configs["num_gpus"] = 1
configs["dev"] = '/gpu:1'  #'/cpu:0'
# configs["dev"] = '/cpu:0'  #'/cpu:0'
configs['tensorboard_on'] = True
configs['tensorboard_refresh'] = 50
#************************************evaluate parameters********************************************
configs['test_num'] = 20

