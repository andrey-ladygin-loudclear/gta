import tensorflow as tf
import numpy as np
import network.cnn_network as cnn

from network.utils import batch_features_labels

X_train = np.load('data/features.npy')
Y_train = np.load('data/labels.npy')

print('Total shape X', X_train.shape)
print('Total shape Y', Y_train.shape)


config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'

imw = 189
imh = 252
n_classes = 3
epochs = 40
batch_size = 256
keep_probability = 0.5

tf.reset_default_graph()

# Inputs
x = cnn.neural_net_image_input((imw, imh, 3))
y = cnn.neural_net_label_input(n_classes)
keep_prob = cnn.neural_net_keep_prob_input()

'''
TensorFlow provides graph collections that group the variables.
To access the variables that were trained you would call
tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
or to get all variables (including some for statistics) use tf.get_collection(tf.GraphKeys.VARIABLES)
or its shorthand tf.all_variables()
'''

# Model
#nn = create_conv2d(X, 128, strides=[8,8], w_name='W1')
w_size, c_strides = cnn.get_weights_shape(x, 32, [8, 8])
W1 = tf.get_variable('W1', w_size, initializer=tf.contrib.layers.xavier_initializer(seed=0))
Z1 = tf.nn.conv2d(x, W1, strides=c_strides, padding='SAME', name='W1_conv2d')
layer = tf.nn.relu(Z1)
max_pool1 = tf.nn.max_pool(layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



#nn = create_conv2d(nn, 256, strides=[4,4], w_name='W2')
w_size, c_strides = cnn.get_weights_shape(max_pool1, 256, [4, 4])
W2 = tf.get_variable('W2', w_size, initializer=tf.contrib.layers.xavier_initializer(seed=0))
Z2 = tf.nn.conv2d(max_pool1, W2, strides=c_strides, padding='SAME', name='W1_conv2d')
layer = tf.nn.relu(Z2)
max_pool2 = tf.nn.max_pool(layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

# nn = create_conv2d(nn, 256, strides=[3,3], w_name='W3')
w_size, c_strides = cnn.get_weights_shape(max_pool2, 256, [3, 3])
W3 = tf.get_variable('W3', w_size, initializer=tf.contrib.layers.xavier_initializer(seed=0))
Z3 = tf.nn.conv2d(max_pool2, W3, strides=c_strides, padding='SAME', name='W1_conv2d')
layer = tf.nn.relu(Z3)
max_pool3 = tf.nn.max_pool(max_pool3, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

nn = create_conv2d(nn, 512, strides=[3,3], w_name='W4')
nn = tf.nn.relu(nn)
nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

nn = create_conv2d(nn, 512, strides=[3,3], w_name='W5')
nn = tf.nn.relu(nn)
layer = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
tf.nn.dropout(layer, keep_prob=keep_prob)

#layer = flatten(layer)
layer = tf.contrib.layers.flatten(layer)
#layer = fully_conn(layer, 400)
layer = tf.contrib.layers.fully_connected(layer, 2000)
layer = tf.nn.dropout(layer, keep_prob)
layer = tf.contrib.layers.fully_connected(layer, 1000)
layer = tf.nn.dropout(layer, keep_prob)

logits = tf.contrib.layers.fully_connected(layer, 3, activation_fn=None)
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