import tensorflow as tf
import network.cnn_network as cnn

n = 1

def make_logits(tensor):
    global n
    # Inputs
    keep_prob = cnn.neural_net_keep_prob_input()

    # Model
    nn = cnn.create_conv2d(tensor, 64, strides=[4, 4], w_name='W1'+str(n))
    nn = tf.nn.relu(nn)
    nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    nn = cnn.create_conv2d(nn, 128, strides=[3, 3], w_name='W2'+str(n))
    nn = tf.nn.relu(nn)
    nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

    nn = cnn.create_conv2d(nn, 256, strides=[2, 2], w_name='W3'+str(n))
    nn = tf.nn.relu(nn)
    nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

    tf.nn.dropout(nn, keep_prob=keep_prob)

    layer = tf.contrib.layers.flatten(nn)
    layer = tf.contrib.layers.fully_connected(layer, 1000)
    n += 1
    return layer