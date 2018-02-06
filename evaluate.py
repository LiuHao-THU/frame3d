import tensorflow as tf
from configs import configs
import densenet121 as model
from read_data import read_data
import matplotlib.pyplot as plt
import six
import scipy.misc
import numpy as np

# read data
read = read_data(root_dir = configs['root_dir'], save_dir = configs['save_dir'], image_size = configs['image_size'], per = configs['per'])
read.build_test_input()

#build the model z
x = read.test_images.astype('float32')[0:16]
y = read.test_labels.astype('int64')[0:16]
eval_images = read.test_images[0:32]
eval_labels = read.test_labels[0:32].astype('uint8')

resnet_model = model.ResNet(ResNet_npy_path = configs['model_path'])
# (self, rgb, label_num, train_mode=None, last_layer_type = "softmax")
resnet_model.build(rgb = x, label_num = configs['num_classes'], train_mode = False, last_layer_type = configs['final_layer_type'])
saver = tf.train.Saver()
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
tf.train.start_queue_runners(sess)
#restore checkpoints
saver.restore(sess, '/home/lh/Desktop/pretrained/resnet_fcn/densenet_121/model/model.ckpt-3192.data-00000-of-00001')
print('model restored')
total_prediction, correct_prediction = 0, 0
for i in six.moves.range(configs['eval_batch_size']):
    loss,prediction = sess.run([net.loss, net.y_soft])
    print(prediction.shape)
    for j in range(prediction.shape[0]):
        index_now = str(j)
        full_name_raw = 'predicted' + '/q' + index_now + '.jpg'
        full_name =  'predicted' + '/' + index_now + '.jpg'
        full_name_labels =  'predicted' + '/p' + index_now + '.jpg'
        scipy.misc.imsave(full_name, prediction[j])
        scipy.misc.imsave(full_name_raw, eval_images[j])
        scipy.misc.imsave(full_name_labels, eval_labels[j])
        eval_labels
        plt.imshow(eval_images[j])
        plt.pause(0.3)

    print('predict finished')
    # tf.logging.info('loss: %.3f, precision: %.3f, best precision: %.3f' % (loss, precision, best_precision))
    if configs["eval_once"]:
        break
# time.sleep(60)
