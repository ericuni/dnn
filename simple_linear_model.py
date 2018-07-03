#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

print(tf.__version__)

# data set
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST/", one_hot = True)
## 每个label 是一个10 个元素的vector, eg: [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.] 将 7 点亮了, 因此label 是 7

print('data set brief:')
print('train set:', data.train.images.shape, data.train.labels.shape)
print('test set:', data.test.images.shape, data.test.labels.shape)
print('validation set:', data.validation.images.shape, data.validation.labels.shape)

## 为了方便之后的比较, 把hot vector 转换为一个数字
data.test.cls = np.array([np.argmax(label) for label in data.test.labels])
print(data.test.cls[0:5])

# model definition
img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_classes = 10 ## 0 - 9 共10 个数字

x = tf.placeholder(tf.float32, [None, img_size_flat]) ## num * 784
y_true = tf.placeholder(tf.float32, [None, num_classes]) ## num * 10
y_true_cls = tf.placeholder(tf.int64, [None]) ## num * 1

weights = tf.Variable(tf.zeros([img_size_flat, num_classes])) ## 784 * 10
bias = tf.Variable(tf.zeros([num_classes])) ## 10 * 1

logits = tf.matmul(x, weights) + bias
y_pred = tf.nn.softmax(logits)
y_pred_cls = tf.argmax(y_pred, axis = 1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = y_true) ## ## 对logits 取softmax, 然后和 y_true 计算cross entropy
cost = tf.reduce_mean(cross_entropy) ## 对所有实例的 cross entropy 求平均
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.5).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# model train
batch_size = 100
session = tf.Session()
session.run(tf.global_variables_initializer())
for i in range(100):
    x_batch, y_batch = data.train.next_batch(batch_size)
    feed_dict_train = {x: x_batch, y_true: y_batch}
    session.run(optimizer, feed_dict = feed_dict_train)

# evaluation
feed_dict_test = {x: data.test.images, y_true_cls: data.test.cls}
acc = session.run(accuracy, feed_dict = feed_dict_test)
print('Accuracy on test set:', acc)

