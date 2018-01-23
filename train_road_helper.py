import tensorflow as tf
import network.cnn_network as cnn

n = 0

def make_logits(tensor, keep_prob):
    global n
    n += 1
    # Inputs
    # keep_prob = cnn.neural_net_keep_prob_input()

    # Model
    nn = cnn.create_conv2d(tensor, 128, strides=[32, 32], w_name='W1'+str(n))
    nn = tf.nn.relu(nn)
    nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    nn = cnn.create_conv2d(nn, 256, strides=[16, 16], w_name='W2'+str(n))
    nn = tf.nn.relu(nn)
    nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

    # nn = cnn.create_conv2d(nn, 256, strides=[8, 8], w_name='W3'+str(n))
    # nn = tf.nn.relu(nn)
    # nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
    #
    # nn = cnn.create_conv2d(nn, 512, strides=[8, 8], w_name='W4'+str(n))
    # nn = tf.nn.relu(nn)
    # nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

    # tf.nn.dropout(nn, keep_prob=keep_prob)
    #
    # nn = cnn.create_conv2d(nn, 512, strides=[6, 6], w_name='W5'+str(n))
    # nn = tf.nn.relu(nn)
    # nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
    #
    # tf.nn.dropout(nn, keep_prob=keep_prob)
    #
    # nn = cnn.create_conv2d(nn, 1024, strides=[6, 6], w_name='W6'+str(n))
    # nn = tf.nn.relu(nn)
    # nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

    tf.nn.dropout(nn, keep_prob=keep_prob)
    #
    # nn = cnn.create_conv2d(nn, 1024, strides=[3, 3], w_name='W7'+str(n))
    # nn = tf.nn.relu(nn)
    # nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

    layer = tf.contrib.layers.flatten(nn)
    # layer = tf.contrib.layers.fully_connected(layer, 2048, activation_fn=tf.nn.relu)
    tf.nn.dropout(layer, keep_prob=keep_prob)
    layer = tf.contrib.layers.fully_connected(layer, 1024, activation_fn=tf.nn.relu)
    # layer = tf.contrib.layers.fully_connected(layer, 100, activation_fn=tf.nn.relu)
    layer = tf.contrib.layers.fully_connected(layer, 1, activation_fn=None)
    return layer


def make_test_logits(tensor, keep_prob):
    pass

def make_simple_logits(tensor, keep_prob):
    nn = cnn.create_conv2d(tensor, 64, strides=[4, 4], w_name='W1')
    nn = tf.nn.relu(nn)
    nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    nn = cnn.create_conv2d(nn, 128, strides=[4, 4], w_name='W2')
    nn = tf.nn.relu(nn)
    nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

    nn = cnn.create_conv2d(nn, 256, strides=[2, 2], w_name='W4')
    nn = tf.nn.relu(nn)
    nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

    nn = cnn.create_conv2d(nn, 512, strides=[2, 2], w_name='W5')
    nn = tf.nn.relu(nn)
    nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

    nn = cnn.create_conv2d(nn, 512, strides=[2, 2], w_name='W6')
    nn = tf.nn.relu(nn)
    nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

    nn = cnn.create_conv2d(nn, 1024, strides=[2, 2], w_name='W7')
    nn = tf.nn.relu(nn)
    nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

    tf.nn.dropout(nn, keep_prob=keep_prob)
    layer = tf.contrib.layers.fully_connected(nn, 2048)# activation_fn=tf.nn.relu)
    tf.nn.dropout(layer, keep_prob=keep_prob)
    layer = tf.contrib.layers.fully_connected(layer, 1024)#, activation_fn=tf.nn.relu)
    tf.nn.dropout(layer, keep_prob=keep_prob)
    layer = tf.contrib.layers.fully_connected(layer, 2, activation_fn=None)
    return layer

    tf.nn.dropout(nn, keep_prob=keep_prob)

    layer = tf.contrib.layers.flatten(nn)
    tf.nn.dropout(layer, keep_prob=keep_prob)
    layer = tf.contrib.layers.fully_connected(layer, 1024, activation_fn=tf.nn.relu)
    layer = tf.contrib.layers.fully_connected(layer, 2, activation_fn=None)
    return layer


def make_logits_for_live_prediction(tensor, keep_prob):
    nn = cnn.create_conv2d(tensor, 64, strides=[4, 4], w_name='W1')
    nn = tf.nn.relu(nn)
    nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    nn = cnn.create_conv2d(nn, 128, strides=[4, 4], w_name='W2')
    nn = tf.nn.relu(nn)
    nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

    nn = cnn.create_conv2d(nn, 256, strides=[2, 2], w_name='W4')
    nn = tf.nn.relu(nn)
    nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

    nn = cnn.create_conv2d(nn, 512, strides=[2, 2], w_name='W5')
    nn = tf.nn.relu(nn)
    nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

    # nn = cnn.create_conv2d(nn, 512, strides=[2, 2], w_name='W6')
    # nn = tf.nn.relu(nn)
    # nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
    #
    # nn = cnn.create_conv2d(nn, 1024, strides=[2, 2], w_name='W7')
    # nn = tf.nn.relu(nn)
    # nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

    tf.nn.dropout(nn, keep_prob=keep_prob)
    layer = tf.contrib.layers.fully_connected(nn, 1024)# activation_fn=tf.nn.relu)
    tf.nn.dropout(layer, keep_prob=keep_prob)
    layer = tf.contrib.layers.fully_connected(layer, 512)#, activation_fn=tf.nn.relu)
    tf.nn.dropout(layer, keep_prob=keep_prob)
    layer = tf.contrib.layers.fully_connected(layer, 1, activation_fn=None)
    return layer