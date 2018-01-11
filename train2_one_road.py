import tensorflow as tf
import numpy as np
import network.cnn_network as cnn

from network.utils import batch_features_labels

X_train = np.load('data/features.npy') / 255
Y_train = np.load('data/labels.npy')

print(X_train.shape)
print(Y_train.shape)

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'

imw = 189
imh = 252
n_classes = 3
epochs = 40
batch_size = 32
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
        sum += session.run(accuracy, feed_dict={x: X_train[i:i+batch_size], y: Y_train[i:i+batch_size], keep_prob: 1.0})

    #validation_accuracy = session.run(accuracy, feed_dict={x: X_train, y: Y_train, keep_prob: 1.0})
    print('Cost = {0} - Validation Accuracy = {1}'.format(cost, sum / count))



save_model_path = 'weights/gta_one_road'
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