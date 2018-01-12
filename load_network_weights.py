import tensorflow as tf
import numpy as np
import network.cnn_network as cnn

X_train = np.load('data/features.npy')
Y_train = np.load('data/labels.npy')

imw = 189
imh = 252
n_classes = 3
epochs = 25
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

saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     # Restore variables from disk.
#     saver.restore(sess, "gta5_weights_v1/gta")
#     print("Model restored.")
#     validation_accuracy = sess.run(accuracy, feed_dict={x: X_train, y: Y_train, keep_prob: 1.0})
#     print("Validation accuracy:", validation_accuracy)

session_conf = tf.ConfigProto(
    device_count={'CPU' : 1, 'GPU' : 0},
    allow_soft_placement=True,
    log_device_placement=False
)
sess = tf.Session(config=session_conf)
saver.restore(sess, "weights/gta_one_road")

print(sum(Y_train))
print(tf.all_variables())



def predict(image):
    soft = tf.nn.softmax(logits)
    prediction = sess.run(soft, {x:image, keep_prob: 1.0})
    #print('prediction', prediction[0], prediction[0] > 0.5)
    print(prediction[0] > 0.5)
    return prediction[0] > 0.5

def getSession():
    with tf.Session() as sess:
        saver.restore(sess, "gta5_weights_v1/gta")
        return sess