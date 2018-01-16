import os
from random import randint
from urllib.request import urlretrieve
from os.path import isfile, isdir

from tqdm import tqdm
import tensorflow as tf
import numpy as np
import network.cnn_network as cnn
#from network import YOLO
from network import models

# for file in os.listdir('data'):
#     print(os.path.join('data', file))
#     # f = h5py.File(os.path.join('data', file), 'r')
#     f = h5py.File(os.path.join('data', file), 'r')
#
#     # print('Keys', list(f.keys()))
#     # print('Values', list(f.values()))
#
#     X_train_dataset = f.get('X_train')
#     Y_train_dataset = f.get('Y_train')
#
#     X_train = np.array(X_train_dataset)
#     Y_train = np.array(Y_train_dataset)
#     print(file, "X_train shape", X_train.shape, "Y_train shape", Y_train.shape)
#     print("")
from network.utils import batch_features_labels, get_train_data

# YOLO.predict()
# raise EOFError

X_train = np.load('data/features.npy')
Y_train = np.load('data/labels.npy')

nx = []
ny = []

print('old: ', X_train.shape)
print('old: ', Y_train.shape)

for x, y in zip(X_train, Y_train):
    if y[0] == 0:
        nx.append(x)
        ny.append([y[1], y[2]])
X_train = np.array(nx)
Y_train = np.array(ny)

print(X_train.shape)
print(Y_train.shape)


config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'

imw = 189#600 #189
imh = 252#800 #252
n_classes = 2
epochs = 25
batch_size = 32
keep_probability = 0.5

tf.reset_default_graph()

# Inputs
x = cnn.neural_net_image_input((imw, imh, 3))
y = cnn.neural_net_label_input(n_classes)
keep_prob = cnn.neural_net_keep_prob_input()

# Model
# nn1 = cnn.create_conv2d(x, 32, strides=[8, 8], w_name='W1')
w_size, c_strides = cnn.get_weights_shape(x, 32, [8, 8])
W1 = tf.get_variable('W1', w_size, initializer=tf.contrib.layers.xavier_initializer(seed=0))
Z1 = tf.nn.conv2d(x, W1, strides=c_strides, padding='SAME', name='W1_conv2d')

nn2 = tf.nn.relu(Z1)
nn3 = tf.nn.max_pool(nn2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
nn = cnn.create_conv2d(nn3, 64, strides=[4, 4], w_name='W2')
nn = tf.nn.relu(nn)
nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
nn = cnn.create_conv2d(nn, 128, strides=[3, 3], w_name='W3')
nn = tf.nn.relu(nn)
nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
nn = cnn.create_conv2d(nn, 256, strides=[3, 3], w_name='W4')
nn = tf.nn.relu(nn)
nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
nn = cnn.create_conv2d(nn, 512, strides=[3, 3], w_name='W5')
nn = tf.nn.relu(nn)
nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
tf.nn.dropout(nn, keep_prob=keep_prob)
layer = tf.contrib.layers.flatten(nn)
layer = tf.contrib.layers.fully_connected(layer, 200)
layer = tf.nn.dropout(layer, keep_prob)
logits = tf.contrib.layers.fully_connected(layer, 2, activation_fn=None)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')


def print_stats(session, feature_batch, label_batch, cost, accuracy):
    cost = session.run(cost, feed_dict={x: feature_batch, y: label_batch, keep_prob: 1.0})
    validation_accuracy = session.run(accuracy, feed_dict={x: X_train, y: Y_train, keep_prob: 1.0})
    print('Cost = {0} - Validation Accuracy = {1}'.format(cost, validation_accuracy))
    test = feature_batch[0]

    # with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        # w1 = tf.get_variable('W1')
        # first_conv2d = tf.get_variable('W1_conv2d')
        # print(w1)
        # print(first_conv2d)


    #var = [v for v in tf.trainable_variables() if v.name == "tower_2/filter:0"][0]
    #print(nn1 * test)
    print(Z1[0])

    #valid_logits = session.run(logits, feed_dict={x: feature_batch, y: label_batch, keep_prob: 1.0})
    #print('Logits', valid_logits, valid_logits.shape, 'feature_batch shape', feature_batch.shape)
    # cost_cross_entropy = session.run(cross_entropy, feed_dict={x: feature_batch, y: label_batch, keep_prob: 1.0})
    # print(cost_cross_entropy)


# print('Checking the Training on a Single Batch...')
# with tf.Session() as sess:
#     # Initializing the variables
#     sess.run(tf.global_variables_initializer())
#
#     # Training cycle
#     for epoch in range(epochs):
#         batch_i = 1
#         for batch_features, batch_labels in utils.load_preprocess_training_batch(batch_i, batch_size):
#             sess.run(optimizer, feed_dict={x: batch_features, y: batch_labels, keep_prob: keep_probability})
#         print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
#         print_stats(sess, batch_features, batch_labels, cost, accuracy)





save_model_path = 'weights/gta'
#
# if os.path.isdir(save_model_path):
#     raise IsADirectoryError(save_model_path + ' is not exists!!')

print('Training...')

with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    # tf.get_variable_scope().reuse_variables()

    # Training cycle
    for epoch in range(epochs):
        for batch_features, batch_labels in batch_features_labels(X_train, Y_train, batch_size):
            sess.run(optimizer, feed_dict={x: batch_features, y: batch_labels, keep_prob: keep_probability})
        print('Epoch {:>2}, CIFAR-10 Batch:  '.format(epoch + 1), end='')
        print_stats(sess, batch_features, batch_labels, cost, accuracy)

    #print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='my_scope'))

    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)