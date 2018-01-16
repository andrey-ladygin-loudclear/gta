import tensorflow as tf
import numpy as np
import network.cnn_network as cnn

from network.utils import batch_features_labels

X_train = np.load('data/features.npy')
Y_train = np.load('data/labels.npy')

#print('sum labels', sum(Y_train))
# [ 1363  1305 13463]

# Epoch 1, Cost = 0.431162029504776 - Train Accuracy = 0.7997448979591837, Test Accuracy = 0.8299999833106995
# Epoch 2, Cost = 0.42498403787612915 - Train Accuracy = 0.7997448979591837, Test Accuracy = 0.8299999833106995
# Epoch 3, Cost = 0.42137521505355835 - Train Accuracy = 0.7997448979591837, Test Accuracy = 0.8299999833106995
# Epoch 4, Cost = 0.4272606074810028 - Train Accuracy = 0.7997448979591837, Test Accuracy = 0.8299999833106995
# Epoch 5, Cost = 0.42544811964035034 - Train Accuracy = 0.7997448979591837, Test Accuracy = 0.8299999833106995

# Epoch 1, Cost = 0.41652974486351013 - Train Accuracy = 0.7997448979591837, Test Accuracy = 0.8299999833106995
# Epoch 2, Cost = 0.42681559920310974 - Train Accuracy = 0.7997448979591837, Test Accuracy = 0.8299999833106995
# Epoch 2, Cost = 0.42775529623031616 - Train Accuracy = 0.7997448979591837, Test Accuracy = 0.8299999833106995

nx = []
ny = []
c = 0
for i in range(len(Y_train)):
    if Y_train[i][0] == 1:
        nx.append(X_train[i])
        ny.append(Y_train[i])
    if Y_train[i][1] == 1:
        nx.append(X_train[i])
        ny.append(Y_train[i])
    if Y_train[i][2] == 1 and c < 1350:
        c += 1
        nx.append(X_train[i])
        ny.append(Y_train[i])

X_train = np.array(nx)
Y_train = np.array(ny)

s = np.arange(X_train.shape[0])
np.random.shuffle(s)

X_train = X_train[s]
Y_train = Y_train[s]

print(sum(Y_train))

# SHOW IAMGES WITH LABELS
# import matplotlib.pyplot as plt
# for im, y in zip(X_train, Y_train):
#     print(y)
#     plt.imshow(im)
#     plt.show()
#
# raise EOFError



print('Total shape X', X_train.shape)
print('Total shape Y', Y_train.shape)
# raise EOFError
# X_test = X_train[0:2000]
# Y_test = Y_train[0:2000]
#
# X_train = X_train[2000:]
# Y_train = Y_train[2000:]
#
# print('X_train shape', X_train.shape)
# print('Y_train shape', Y_train.shape)
#
# print('X_test shape', X_test.shape)
# print('Y_test shape', Y_test.shape)

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'

imw = 189
imh = 252
n_classes = 3
epochs = 2
batch_size = 64
keep_probability = 0.5

tf.reset_default_graph()

# Inputs
x = cnn.neural_net_image_input((imw, imh, 3))
y = cnn.neural_net_label_input(n_classes)
keep_prob = cnn.neural_net_keep_prob_input()

# Model
logits = cnn.conv_net(x, keep_prob)
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

    sum = 0
    count = len(X_train) // batch_size

    for i in range(count):
        sum += session.run(accuracy, feed_dict={x: X_train[i:i+batch_size] / 255, y: Y_train[i:i+batch_size], keep_prob: 1.0})

    test_accuracy = 0#session.run(accuracy, feed_dict={x: X_test / 255, y: Y_test, keep_prob: 1.0})

    print('Cost = {0} - Train Accuracy = {1}, Test Accuracy = {2}'.format(cost, sum / count, test_accuracy))



save_model_path = 'weights/gta_one_road'
print('Training...')

with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(epochs):
        for batch_features, batch_labels in batch_features_labels(X_train, Y_train, batch_size):
            sess.run(optimizer, feed_dict={x: batch_features / 255, y: batch_labels, keep_prob: keep_probability})
        print('Epoch {:>2}, '.format(epoch + 1), end='')
        print_stats(sess, batch_features, batch_labels, cost, accuracy)

    #print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='my_scope'))

    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)