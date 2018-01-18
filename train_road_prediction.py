import cv2
import tensorflow as tf
import numpy as np
import network.cnn_network as cnn
from network.utils import batch_features_labels, triplet_loss, batch_features_labels_triple
from train_road_helper import make_logits

X_road_train = np.load('data/straights_road_features.npy')
Y_non_road_train = np.load('data/non_road_features.npy')

print(X_road_train.shape)
print(Y_non_road_train.shape)
print('Make the same size')

N = 32

Anchor_train = X_road_train[N:N*2]
X_road_train = X_road_train[:N]
Y_non_road_train = Y_non_road_train[:N]

print(Anchor_train.shape)
print(X_road_train.shape)
print(Y_non_road_train.shape)

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'

imw = 90
imh = 120
n_classes = 1
epochs = 500
batch_size = 32
keep_probability = 0.5

tf.reset_default_graph()


anchor_image = cnn.neural_net_image_input((imw, imh, 3), name='anchor_image')
positive_image = cnn.neural_net_image_input((imw, imh, 3), name='positive_image')
negative_image = cnn.neural_net_image_input((imw, imh, 3), name='negative_image')
keep_prob = cnn.neural_net_keep_prob_input()

anchor = make_logits(anchor_image, keep_prob)
positive = make_logits(positive_image, keep_prob)
negative = make_logits(negative_image, keep_prob)

# Loss and Optimizer
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
# cost = tf.reduce_mean(cross_entropy)
# loss = triplet_loss(y_true, y_pred)
loss = triplet_loss(anchor, positive, negative)
optimizer = tf.train.AdamOptimizer().minimize(loss)

# Accuracy
#correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
# correct_pred = tf.equal(logits, y)
#correct_pred = abs(logits - y)
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

#encoding = img_to_encoding(image_path, model)
#dist = np.linalg.norm(database[identity] - encoding)


def img_to_encoding(image_path, model):
    img1 = cv2.imread(image_path, 1)
    img = img1[...,::-1]
    img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding

save_model_path = 'weights/gta_check_road'
print('Training...')

with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(epochs):
        batches_count = 0
        cost_sum = 0

        for batch_anchor_image, batch_X_road, batch_Y_non_road in batch_features_labels_triple(Anchor_train, X_road_train, Y_non_road_train, batch_size):
            _, cost = sess.run([optimizer, loss], feed_dict={
                anchor_image: batch_anchor_image / 255,
                positive_image: batch_X_road / 255,
                negative_image: batch_Y_non_road / 255,
                keep_prob: keep_probability})
            batches_count +=1
            cost_sum += cost

            #print('Batch: ', batches_count, 'Cost: ', cost)

        print('Epoch {:>2}, Batch:  '.format(epoch + 1), end='')
        print('Cost: ', (cost_sum / batches_count))
        #print_stats(sess, batch_features / 255, batch_labels, cost, accuracy)

    #print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='my_scope'))

    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)