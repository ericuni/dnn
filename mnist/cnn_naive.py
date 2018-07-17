import numpy as np
import tensorflow as tf
import sys
from mnist_data import *

data = load_data()

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

layer_conv1 = tf.layers.conv2d(inputs=x_image, filters=16, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
layer_pool1 = tf.layers.max_pooling2d(inputs=layer_conv1, pool_size=[2, 2], strides=2)
layer_conv2 = tf.layers.conv2d(inputs=layer_pool1, filters=36, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
layer_pool2 = tf.layers.max_pooling2d(inputs=layer_conv2, pool_size=[2, 2], strides=2)
layer_flat  = tf.reshape(layer_pool2, [-1, 7 * 7 * 36])
layer_dense = tf.layers.dense(inputs=layer_flat, units=1024, activation=tf.nn.relu)
layer_dropout = tf.layers.dropout(inputs=layer_dense, rate=0.4)
logits = tf.layers.dense(inputs=layer_dropout, units=10)

y_pred = tf.nn.softmax(logits)
y_pred_cls = tf.argmax(y_pred, axis=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()
session.run(tf.global_variables_initializer())

batch_size = 64
iterations = 500
feed_dict_test = {x: data.test.images, y_true_cls: data.test.cls}
for i in range(iterations):
	x_batch, y_true_batch = data.train.next_batch(batch_size)
	feed_dict_train = {x: x_batch, y_true: y_true_batch}
	session.run(optimizer, feed_dict=feed_dict_train)
	if i % 10 == 0 or i == iterations - 1:
		acc = session.run(accuracy, feed_dict = feed_dict_test)
		print('accuracy after {0} iterations: {1}'.format(i, acc))

result = session.run(y_pred_cls, feed_dict = {x: data.validation.images[0:19]})
print('true: {}'.format(data.validation.cls[0:19]))
print('pred: {}'.format(result))

