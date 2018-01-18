import tensorflow as tf
import numpy as np
import network.cnn_network as cnn
from network.utils import batch_features_labels, ff_filter


X_train = np.load('data/features.npy')
Y_train = np.load('data/labels.npy')

print(X_train.shape)
print(Y_train.shape)

X_train, Y_train = ff_filter(X_train, Y_train)

print(X_train.shape)
print(Y_train.shape)


config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'

imw = 90
imh = 120
n_classes = 1
epochs = 100
batch_size = 64
keep_probability = 0.5
save_model_path = 'weights/gta1'

tf.reset_default_graph()

# Inputs
x = cnn.neural_net_image_input((imw, imh, 3))
y = cnn.neural_net_label_input(n_classes)
keep_prob = cnn.neural_net_keep_prob_input()

# Model
nn = cnn.create_conv2d(x, 64, strides=[3, 3], w_name='W1')
nn = tf.nn.relu(nn, name='W1_activated')
nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
nn = cnn.create_conv2d(nn, 128, strides=[3, 3], w_name='W2')
nn = tf.nn.relu(nn, name='W2_activated')
nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
nn = cnn.create_conv2d(nn, 256, strides=[2, 2], w_name='W3')
nn = tf.nn.relu(nn, name='W3_activated')
nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
nn = cnn.create_conv2d(nn, 512, strides=[2, 2], w_name='W4')
nn = tf.nn.relu(nn, name='W4_activated')
nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
tf.nn.dropout(nn, keep_prob=keep_prob)
layer = tf.contrib.layers.flatten(nn)
layer = tf.contrib.layers.fully_connected(layer, 1000)
layer = tf.nn.dropout(layer, keep_prob)
layer = tf.contrib.layers.fully_connected(layer, 1000)
layer = tf.nn.dropout(layer, keep_prob)
logits = cnn.output(layer, n_classes)
logits = tf.identity(logits, name='logits')
sq = tf.square(y-logits)
cost = tf.reduce_mean(sq)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
correct_pred = tf.divide(tf.abs(tf.subtract(y, logits)), y)
accuracy = tf.reduce_mean(correct_pred, 0)

session_conf = tf.ConfigProto(
    device_count={'CPU' : 1, 'GPU' : 0},
    allow_soft_placement=True,
    log_device_placement=False
)
saver = tf.train.Saver()
sess = tf.Session(config=session_conf)
saver.restore(sess, save_model_path)

def predict(image):
    #soft = tf.nn.softmax(logits)
    prediction = sess.run(logits, {x:image, keep_prob: 1.0})
    print(prediction)
    return prediction[0]

# import matplotlib.pyplot as plt
# from PIL import Image
# for im in X_train:
#     image = Image.fromarray(im, 'RGB')
#     predict([im])
#     plt.imshow(image)
#     plt.show()