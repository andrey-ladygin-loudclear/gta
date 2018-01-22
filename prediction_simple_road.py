import cv2
import tensorflow as tf
import numpy as np
import network.cnn_network as cnn
from network.utils import batch_features_labels, triplet_loss, batch_features_labels_triple, show_image
from train_road_helper import make_logits, make_simple_logits

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
Total_X = X_train[s]
Total_Y = Y_train[s]

X_train = Total_X[:20000]
Y_train = Total_Y[:20000]
X_dev = Total_X[20000:]
Y_dev = Total_Y[20000:]

#Total 23000
print('X_train', X_train.shape)
print('Y_train', Y_train.shape)
print('X_dev', X_dev.shape)
print('Y_dev', Y_dev.shape)

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

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
# correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

save_model_path = 'weights/gta_simple_road_prediction_sigmoid_4'

session_conf = tf.ConfigProto(
    device_count={'CPU' : 1, 'GPU' : 0},
    allow_soft_placement=True,
    log_device_placement=False
)
saver = tf.train.Saver()
sess = tf.Session(config=session_conf)
saver.restore(sess, save_model_path)

def predict(image):
    prediction = sess.run(logits, {x: image, keep_prob: 1.0})
    print(prediction[0][0][0] > 0.5)
    return prediction[0]

for i in range(20):
    predict([X_dev[i+40] / 255])
    show_image(X_dev[i])