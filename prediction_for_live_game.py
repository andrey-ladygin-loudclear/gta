import cv2
import tensorflow as tf
import numpy as np
import network.cnn_network as cnn
from network.utils import batch_features_labels, triplet_loss, batch_features_labels_triple, show_image
from prediction_simple_road import predict_if_it_is_road
from train_road_helper import make_logits, make_simple_logits, make_logits_for_live_prediction

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'

imw = 90
imh = 120
n_classes = 1
epochs = 200
batch_size = 10
keep_probability = 0.5

tf.reset_default_graph()

x = cnn.neural_net_image_input((imw, imh, 3), name='net')
y = cnn.neural_net_label_input(n_classes)
keep_prob = cnn.neural_net_keep_prob_input()

logits = make_logits_for_live_prediction(x, keep_prob)
logits = tf.identity(logits, name='logits')

# is_road = predict_if_it_is_road(x)
# is_road = is_road[0]

if y[0] == 1.0:
    loss = logits
else:
    loss = logits + 5000

optimizer = tf.train.AdamOptimizer().minimize(loss)

save_model_path = 'weights/gta_live_prediction'

session_conf = tf.ConfigProto(
    device_count={'CPU' : 1, 'GPU' : 0},
    allow_soft_placement=True,
    log_device_placement=False
)
sess = tf.Session(config=session_conf)
sess.run(tf.global_variables_initializer())


def live_train(train_x):
    train_x = np.array(train_x)
    y_train = predict_if_it_is_road(train_x)
    if y_train[0]:
        y_train = [1.0]
    else:
        y_train = [0.0]

    y_train = np.array([y_train])

    print('is road', y_train, y_train.shape, 'train_x', train_x.shape)

    sess.run(optimizer, feed_dict={x: train_x, y: y_train, keep_prob: keep_probability})
    #sess.run(optimizer, feed_dict={x: train_x, y: train_y, keep_prob: keep_probability})


def live_prediction(train_x):
    l, c = sess.run([logits, loss], feed_dict={x: train_x, keep_prob: 1.0})
    print('Lost', l, 'Cost', c)
    return l

    # Save Model
    # saver = tf.train.Saver()
    # save_path = saver.save(sess, save_model_path)