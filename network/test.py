from PIL import Image
import numpy as np
from scipy import ndimage
import cnn_network as cnn
import tensorflow as tf
import models

fname = "test.png"
real_image = ndimage.imread(fname)
image = np.array(real_image)
print(image.shape)


tf.reset_default_graph()

imw = 800
imh = 600
n_classes = 4
epochs = 100
batch_size = 64
keep_probability = 0.5

# Inputs
x = cnn.neural_net_image_input((imw, imh, 3))
y = cnn.neural_net_label_input(n_classes)
keep_prob = cnn.neural_net_keep_prob_input()

logits = models.test_net(x, keep_prob)

logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

def print_stats(session, feature_batch, label_batch, cost, accuracy):
    cost = session.run(cost, feed_dict={x: feature_batch, y: label_batch, keep_prob: 1.0})
    validation_accuracy = session.run(accuracy, feed_dict={x: valid_features, y: valid_labels, keep_prob: 1.0})
    print('('+name+'), Cost = {0} - Validation Accuracy = {1}'.format(cost, validation_accuracy))

print('Checking the Training ('+name+') on a Single Batch...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(epochs):
        batch_i = 1
        for batch_features, batch_labels in utils.load_preprocess_training_batch(batch_i, batch_size):
            sess.run(optimizer, feed_dict={x: batch_features, y: batch_labels, keep_prob: keep_probability})
        print('Epoch {:>2}, ('+name+') CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
        print_stats(sess, batch_features, batch_labels, cost, accuracy)