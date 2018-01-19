import cv2
import tensorflow as tf
import numpy as np
import network.cnn_network as cnn
from network.utils import batch_features_labels, triplet_loss, batch_features_labels_triple, show_image
from train_road_helper import make_logits

X_train = np.load('data/straights_road_features.npy')
Y_train = np.load('data/non_road_features.npy')

print(X_train.shape)
print(Y_train.shape)
print('Make the same size')


road = np.array([X_train[1]])
non_road = np.array([Y_train[1]])



#N = 32*60
N = 10*80


positive_batch = []
negative_batch = []
import random
for i in range(N):
    positive_batch.append(random.choice(X_train))
    negative_batch.append(random.choice(Y_train))

positive_batch = np.array(positive_batch)
negative_batch = np.array(negative_batch)


Anchor_train = X_train[N:N*2]
X_road_train = X_train
Y_non_road_train = Y_train


config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'

imw = 90
imh = 120
n_classes = 1
epochs = 5
batch_size = 10
keep_probability = 0.5

tf.reset_default_graph()

anchor_image = cnn.neural_net_image_input((imw, imh, 3), name='anchor_image')
positive_image = cnn.neural_net_image_input((imw, imh, 3), name='positive_image')
negative_image = cnn.neural_net_image_input((imw, imh, 3), name='negative_image')
keep_prob = cnn.neural_net_keep_prob_input()

anchor = make_logits(anchor_image, keep_prob)
positive = make_logits(positive_image, keep_prob)
negative = make_logits(negative_image, keep_prob)


loss = triplet_loss(anchor, positive, negative, alpha=.5)
optimizer = tf.train.AdamOptimizer().minimize(loss)



def img_to_encoding(image_path, model):
    img1 = cv2.imread(image_path, 1)
    img = img1[...,::-1]
    img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding


def check_dist(sess):
    c = 400
    dist1 = 0
    dist2 = 0

    encodecRoad = sess.run(anchor, feed_dict={anchor_image: road / 255})
    encodecNonRoad = sess.run(anchor, feed_dict={anchor_image: non_road / 255})

    for i in range(c):
        check_road = np.array([X_train[2000 + i]])
        check_non_road = np.array([Y_train[1500 + i]])
        encodec1 = sess.run(anchor, feed_dict={anchor_image: check_road / 255})
        encodec2 = sess.run(anchor, feed_dict={anchor_image: check_non_road / 255})
        dist1 = np.linalg.norm(encodecRoad - encodec1)
        dist2 = np.linalg.norm(encodecRoad - encodec2)

    print('Dist Road', (dist1 / c), 'AND', (dist2 / c))
    # print('Dist NON Road', np.linalg.norm(encodecNonRoad - encodec1), np.linalg.norm(encodecNonRoad - encodec2))

save_model_path = 'weights/gta_check_road'
print('Training...')

with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(epochs):
        batches_count = 0
        cost_sum = 0

        #for batch_anchor_image, batch_X_road, batch_Y_non_road in batch_features_labels_triple(Anchor_train, X_road_train, Y_non_road_train, batch_size):
        for positive_batch_train, negative_batch_train in batch_features_labels(positive_batch, negative_batch, batch_size):
            _, cost = sess.run([optimizer, loss], feed_dict={
                anchor_image: road / 255,
                positive_image: positive_batch_train / 255,
                negative_image: negative_batch_train / 255,
                keep_prob: keep_probability})
            batches_count +=1
            cost_sum += cost

            # _, cost = sess.run([optimizer, loss], feed_dict={
            #     anchor_image: non_road / 255,
            #     positive_image: batch_Y_non_road / 255,
            #     negative_image: batch_X_road / 255,
            #     # positive_image: batch_X_road / 255,
            #     # negative_image: batch_Y_non_road / 255,
            #     keep_prob: keep_probability})
            # batches_count +=1
            # cost_sum += cost

            #print('Batch: ', batches_count, 'Cost: ', cost)

        print('Epoch {:>2}, Batch:  '.format(epoch + 1), end='')
        print('Cost: ', (cost_sum / batches_count), end=' ')
        check_dist(sess)
        # if cost_sum == 0:
        #     print('Cost is minimized. Break')
        #     break
        #print_stats(sess, batch_features / 255, batch_labels, cost, accuracy)

    #print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='my_scope'))

    encodec1 = sess.run(anchor, feed_dict={anchor_image: road / 255})

    for i in range(5000, 5015):
        check = np.array([X_train[i]])
        encodec2 = sess.run(anchor, feed_dict={anchor_image: check / 255})
        dist = np.linalg.norm(encodec1 - encodec2)
        print(dist)
        show_image(X_train[i])

    for i in range(1850, 1865):
        check = np.array([Y_train[i]])
        encodec2 = sess.run(anchor, feed_dict={anchor_image: check / 255})
        dist = np.linalg.norm(encodec1 - encodec2)
        print(dist)
        show_image(Y_train[i])

    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)