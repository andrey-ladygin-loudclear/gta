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

print(X_train.shape)
print(Y_train.shape)
# print(Y_train)

# raise ValueError

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'

imw = 189#600 #189
imh = 252#800 #252
n_classes = 2
epochs = 58
batch_size = 32
keep_probability = 0.5

tf.reset_default_graph()

#Epoch 25, CIFAR-10 Batch:  Cost = 0.42823734879493713 - Validation Accuracy = 0.7324798107147217

# Inputs
x = cnn.neural_net_image_input((imw, imh, 3))
y = cnn.neural_net_label_input(n_classes)
keep_prob = cnn.neural_net_keep_prob_input()

# Model
#https://www.coursera.org/learn/convolutional-neural-networks/lecture/GboGx/what-are-deep-convnets-learning
logits = cnn.conv_net(x, keep_prob)

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