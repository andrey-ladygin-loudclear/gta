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


check_X = []
for item in features[3000:5000]: check_X.append(item)
for item in non_road_train: check_X.append(item)

check_X = np.array(check_X)

s = np.arange(check_X.shape[0])
np.random.shuffle(s)
check_X = check_X[s]
# print('check_X', check_X.shape)
# show_image(check_X[1])
# raise EOFError

X_dev = []
Y_dev = []
for item in features[4000:5000]:
    X_dev.append(item)
    Y_dev.append([1, 0])
for item in non_road_train[:1000]:
    X_dev.append(item)
    Y_dev.append([0, 1])
X_dev = np.array(X_dev)
Y_dev = np.array(Y_dev)

for item in features[:2000]:
    X_train.append(item)
    Y_train.append([1, 0])

for item in road_train[:4000]:
    X_train.append(item)
    Y_train.append([1, 0])

for item in non_road_train:
    X_train.append(item)
    Y_train.append([0, 1])


X_train = np.array(X_train)
Y_train = np.array(Y_train)

s = np.arange(X_train.shape[0])
np.random.shuffle(s)
Total_X = X_train[s]
Total_Y = Y_train[s]

print('Total features:', len(X_train))

X_train = Total_X
Y_train = Total_Y
# X_dev = X_train[18000:19000]
# Y_dev = Y_train[18000:19000]

print('X_train', X_train.shape)
print('Y_train', Y_train.shape, sum(Y_train))
print('X_dev', X_dev.shape)
print('Y_dev', Y_dev.shape)

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'

imw = 90
imh = 120
n_classes = 2
epochs = 7
batch_size = 256
keep_probability = 0.5

tf.reset_default_graph()

x = cnn.neural_net_image_input((imw, imh, 3), name='net')
y = cnn.neural_net_label_input(n_classes)
keep_prob = cnn.neural_net_keep_prob_input()

logits = make_simple_logits(x, keep_prob)
# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)
# optimizer = tf.train.GradientDescentOptimizer(3).minimize(cost)

# Accuracy
p = logits[0][0]
correct_pred = tf.equal(tf.argmax(p, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')


save_model_path = 'weights/gta_simple_road_prediction_sigmoid_4'
print('Training...')

with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(epochs):
        batches_count = 0
        dev_batches_count = 0
        cost_sum = 0
        accuracy_sum = 0
        dev_cost_sum = 0
        dev_accuracy_sum = 0

        for x_batch, y_batch in batch_features_labels(X_train, Y_train, batch_size):
            sess.run(optimizer, feed_dict={x: x_batch / 255, y: y_batch, keep_prob: keep_probability})

        for x_batch, y_batch in batch_features_labels(X_train, Y_train, batch_size):
            c, a, pred = sess.run([cost, accuracy, p], feed_dict={x: x_batch / 255, y: y_batch, keep_prob: 1.0})
            batches_count += 1
            cost_sum += c
            accuracy_sum += a

            # for i in range(len(pred)):
            #     print('Check', pred[i], y_batch[i])

        for x_batch, y_batch in batch_features_labels(X_dev, Y_dev, batch_size):
            c, a = sess.run([cost, accuracy], feed_dict={x: x_batch / 255, y: y_batch, keep_prob: 1.0})
            dev_batches_count += 1
            dev_cost_sum += c
            dev_accuracy_sum += a

        print('Epoch {:>2}, '.format(epoch + 1), end='')
        print('Cost: ', (cost_sum / batches_count), 'Accuracy: ', (accuracy_sum / batches_count), 'Dev cost: ', (dev_cost_sum / dev_batches_count), 'Dev accuracy: ', (dev_accuracy_sum / dev_batches_count))
        # print('Cost: ', (cost_sum / batches_count), 'Dev cost: ', (dev_cost_sum / dev_batches_count))

        if epoch + 1 >= 1000 and (cost_sum / batches_count) < 1:
            break

    # for i in check_X:
    #     a = sess.run(logits, feed_dict={x: [i / 255], keep_prob: 1.0})
    #     print(a)
    #     show_image(i)


        # Save Model
    print('Saving model as', save_model_path)
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)