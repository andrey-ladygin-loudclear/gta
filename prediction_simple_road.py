import cv2
import tensorflow as tf
import numpy as np
import network.cnn_network as cnn
from network.utils import batch_features_labels, triplet_loss, batch_features_labels_triple, show_image
from train_road_helper import make_logits, make_simple_logits

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

logits = make_simple_logits(x, keep_prob)
logits = tf.identity(logits, name='logits')

loss = tf.reduce_mean(tf.pow(logits-y, 2))
optimizer = tf.train.AdamOptimizer().minimize(loss)

save_model_path = 'weights/gta_simple_road_prediction'
#save_model_path = 'weights/gta_simple_road_prediction'

session_conf = tf.ConfigProto(
    device_count={'CPU' : 1, 'GPU' : 0},
    allow_soft_placement=True,
    log_device_placement=False
)
saver = tf.train.Saver()
sess = tf.Session(config=session_conf)
saver.restore(sess, save_model_path)

def predict(image):
    prediction = sess.run(logits, {x:image, keep_prob: 1.0})
    print(prediction)
    return prediction[0]