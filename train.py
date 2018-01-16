import tensorflow as tf
import numpy as np
import network.cnn_network as cnn
from network.utils import batch_features_labels

l = np.linalg.norm(
    np.array([1,2,3,4,5,6]) -
    np.array([2,5,3,4,5,6])
)
print(l)
raise EOFError

X_train = np.load('data/features.npy')
Y_train = np.load('data/labels.npy')

print(X_train.shape)
print(Y_train.shape)

raise EOFError

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'

imw = 90
imh = 120
n_classes = 1
epochs = 5
batch_size = 32
keep_probability = 0.5

tf.reset_default_graph()

# Inputs
x = cnn.neural_net_image_input((imw, imh, 3))
y = cnn.neural_net_label_input(n_classes)
keep_prob = cnn.neural_net_keep_prob_input()

# Model
nn = cnn.create_conv2d(x, 64, strides=[4, 4], w_name='W1')
nn = tf.nn.relu(nn, name='W1_activated')
nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

nn = cnn.create_conv2d(nn, 128, strides=[3, 3], w_name='W2')
nn = tf.nn.relu(nn, name='W2_activated')
nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

nn = cnn.create_conv2d(nn, 128, strides=[2, 2], w_name='W3')
nn = tf.nn.relu(nn, name='W3_activated')
nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

tf.nn.dropout(nn, keep_prob=keep_prob)

layer = tf.contrib.layers.flatten(nn)
layer = tf.contrib.layers.fully_connected(layer, 500)
layer = tf.nn.dropout(layer, keep_prob)
layer = tf.contrib.layers.fully_connected(layer, 500)
layer = tf.nn.dropout(layer, keep_prob)

logits = tf.contrib.layers.fully_connected(layer, n_classes, activation_fn=None)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
# cost = tf.reduce_mean(cross_entropy)
cost = tf.reduce_mean(tf.square(logits-y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
#correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
# correct_pred = tf.equal(logits, y)
correct_pred = abs(logits - y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')


def print_stats(session, feature_batch, label_batch, cost, accuracy):
    cost = session.run(cost, feed_dict={x: feature_batch, y: label_batch, keep_prob: 1.0})

    sum = 0
    count = len(X_train) // batch_size

    for i in range(count):
        sum += session.run(accuracy, feed_dict={x: X_train[i:i+batch_size] / 255, y: Y_train[i:i+batch_size], keep_prob: 1.0})

    prediction, cpd = session.run([logits, correct_pred], feed_dict={x: feature_batch, y: label_batch, keep_prob: 1.0})

    for i in range(len(prediction)):
        print(prediction[i], label_batch[i], cpd[i])
    print('Cost = {0} - Validation Accuracy = {1}'.format(cost, sum / count))


save_model_path = 'weights/gta1'
print('Training...')

with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(epochs):
        for batch_features, batch_labels in batch_features_labels(X_train, Y_train, batch_size):
            sess.run(optimizer, feed_dict={x: batch_features / 255, y: batch_labels, keep_prob: keep_probability})
        print('Epoch {:>2}, CIFAR-10 Batch:  '.format(epoch + 1), end='')
        print_stats(sess, batch_features / 255, batch_labels, cost, accuracy)

    #print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='my_scope'))

    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)