import cv2
import tensorflow as tf
import numpy as np
import network.cnn_network as cnn
from network.utils import batch_features_labels, triplet_loss, batch_features_labels_triple, show_image
from train_road_helper import make_logits, make_simple_logits

features = np.load('data/features.npy')
road_train = np.load('data/straights_road_features.npy')
non_road_train = np.load('data/non_road_features2.npy')

X_train = []
Y_train = []

print('features', features.shape)
print('road_train', road_train.shape)
print('non_road_train', non_road_train.shape)
print('Make the same size')

for item in features:
    X_train.append(item)
    Y_train.append([1, 0])

for item in road_train:
    X_train.append(item)
    Y_train.append([1, 0])

for item in non_road_train:
    X_train.append(item)
    Y_train.append([0, 1])



X_train = np.array(X_train)
Y_train = np.array(Y_train)

s = np.arange(X_train.shape[0])
np.random.shuffle(s)
X_train = X_train[s]
Y_train = Y_train[s]


print(X_train.shape)
print(Y_train.shape)

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'

imw = 90
imh = 120
n_classes = 2
epochs = 200
batch_size = 10
keep_probability = 0.5

tf.reset_default_graph()

x = cnn.neural_net_image_input((imw, imh, 3), name='net')
y = cnn.neural_net_label_input(n_classes)
keep_prob = cnn.neural_net_keep_prob_input()

logits = make_simple_logits(x, keep_prob)
logits = tf.identity(logits, name='logits')

sigmoid = tf.nn.sigmoid(logits)
#cross_entropy = -tf.reduce_sum(y * tf.log(sigmoid), reduction_indices=1)
#loss = tf.reduce_mean(cross_entropy)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=sigmoid, labels=y)
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(sigmoid * tf.log(y), reduction_indices=[1]))
loss = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer().minimize(loss)

save_model_path = 'weights/gta_simple_road_prediction_sigmoid_4'
print('Training...')

with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(epochs):
        batches_count = 0
        cost_sum = 0

        for x_batch, y_batch in batch_features_labels(X_train, Y_train, batch_size):
            sess.run(optimizer, feed_dict={x: x_batch / 255, y: y_batch, keep_prob: keep_probability})

        for x_batch, y_batch in batch_features_labels(X_train, Y_train, batch_size):
            cost = sess.run(loss, feed_dict={x: x_batch / 255, y: y_batch, keep_prob: keep_probability})
            batches_count += 1
            cost_sum += cost

        print('Epoch {:>2}, '.format(epoch + 1), end='')
        print('Cost: ', (cost_sum / batches_count))

        if epoch > 16 and cost_sum / batches_count < 10:
            break

    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)