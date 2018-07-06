#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3

import logging
import numpy as np
import sklearn as sk
import sys
import tensorflow as tf
import time
import udf

logging.basicConfig(level = logging.DEBUG, format = '%(levelname)s %(asctime)s [%(filename)s][%(lineno)d][%(funcName)s] %(message)s')
log = logging.getLogger()

log.info('tensorflow version: {0}'.format(tf.__version__))

# data set
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST/", one_hot = True)
## 每个label 是一个10 个元素的vector, eg: [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.] 将 7 点亮了, 因此label 是 7

log.info('data set brief:')
log.info('train set: {0} {1}'.format(data.train.images.shape, data.train.labels.shape)) ## (55000, 784) (55000, 10)
log.info('test set: {0} {1}'.format(data.test.images.shape, data.test.labels.shape))
log.info('validation set: {0} {1}'.format(data.validation.images.shape, data.validation.labels.shape))

img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_channels = 1 ## number of colour channel for the images, 1 channel for gray-scale
num_classes = 10 ## 0 - 9 共10 个数字

data.test.cls = np.array([np.argmax(label) for label in data.test.labels])

# model definition
## convolution layer 1
filter_size1 = 5 ## convolution filters are 5 * 5 pixels
num_filters1 = 16

## convolution layer 2
filter_size2 = 5
num_filters2 = 36

## fully connected layer
fc_size = 128

x = tf.placeholder(tf.float32, [None, img_size_flat], name = 'x') ## num * 784
'''
The convolutional layers expect x to be encoded as a 4-dim tensor so we have to reshape it so its shape is instead [num_images, img_height, img_width, num_channels].
-1 means that the number of images can be inferred automatically
'''
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, [None, num_classes]) ## num * 10
y_true_cls = tf.argmax(y_true, axis = 1)

layer_conv1, weights_conv1 = udf.new_conv_layer(input = x_image, num_input_channels = num_channels, filter_size = filter_size1, num_filters = num_filters1, use_pooling = True)
log.info('layer {0}: {1} {2}'.format('convolution layer 1', layer_conv1, weights_conv1))
'''
layer_conv1 with shape=(?, 14, 14, 16) and weights_conv1 with shape=(5, 5, 1, 16)
'''

layer_conv2, weights_conv2 = udf.new_conv_layer(input = layer_conv1, num_input_channels = num_filters1, filter_size = filter_size2, num_filters = num_filters2, use_pooling = True)
log.info('layer {0}: {1} {2}'.format('convolution layer 2', layer_conv2, weights_conv2))
'''
layer_conv1 with shape=(?, 7, 7, 36) and weights_conv1 with shape=(5, 5, 16, 36)
'''

layer_flat, num_features = udf.flatten_layer(layer_conv2)
log.info('layer {0}: {1} {2}'.format('flatten layer', layer_flat, num_features))
## layer_flat with shape=(?, 1764), 1764 = 7 * 7 * 36

layer_fc1 = udf.new_fc_layer(input = layer_flat, num_inputs = num_features, num_outputs = fc_size, use_relu = True)
log.info('layer {0}: {1}'.format('fully connected layer 1', layer_fc1))
## layer_fc1 with shape=(?, 128)

layer_fc2 = udf.new_fc_layer(input = layer_fc1, num_inputs = fc_size, num_outputs = num_classes, use_relu = False)
log.info('layer {0}: {1}'.format('fully connected layer 2', layer_fc2))
## layer_fc2 with shape=(?, 10)

y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, axis = 1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits = layer_fc2, labels = y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# model train
batch_size = 64
iterations = 200
session = tf.Session()
session.run(tf.global_variables_initializer())
feed_dict_test = {x: data.test.images, y_true_cls: data.test.cls}
time_start = time.time()
for i in range(iterations):
	x_batch, y_batch = data.train.next_batch(batch_size)
	feed_dict_train = {x: x_batch, y_true: y_batch}
	session.run(optimizer, feed_dict = feed_dict_train)
	if i % 10 == 0 or i == iterations - 1:
		acc = session.run(accuracy, feed_dict = feed_dict_test)
		log.info('accuracy after {0} iterations: {1}'.format(i, acc))
time_total = time.time() - time_start
log.info('model train takes: %d secs' % time_total)

# evaluation
cls_true = data.test.cls
cls_pred = session.run(y_pred_cls, feed_dict = feed_dict_test)
cm = sk.metrics.confusion_matrix(y_true = cls_true, y_pred = cls_pred)
log.info('Confusion matrix:\n {0}'.format(cm))
udf.plot_confusion_matrix('Confusion matrix', cm, num_classes)

session.close()
sys.exit(0)

