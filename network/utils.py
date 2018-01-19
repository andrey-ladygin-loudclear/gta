import pickle
import os
import h5py
import numpy as np
import tables
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image
import tensorflow as tf

def normalize(features):
    return features/255.

def one_hot_encode(x):
    n = len(x)
    b = np.zeros((n, max(x)+1))
    b[np.arange(n), x] = 1
    return b

def preprocess_and_save_data(cifar10_dataset_folder_path):
    def _save_data(name, save_as=None, start_index=None, end_index=None):
        features, labels = load_cfar10_batch_by_name(cifar10_dataset_folder_path+'/'+name)
        data_len = len(features)

        if start_index and start_index < 1: start_index = int(data_len * start_index)
        if end_index and end_index < 1: end_index = int(data_len * end_index)

        if start_index and end_index:
            features = features[start_index: end_index]
            labels = labels[start_index: end_index]
        elif start_index:
            features = features[start_index:]
            labels = labels[start_index:]
        else:
            features = features[:end_index]
            labels = labels[:end_index]

        features = normalize(features)
        labels = one_hot_encode(labels)
        if not save_as: save_as = str(name) + '.p'
        print(cifar10_dataset_folder_path + '/' + save_as, len(features))
        pickle.dump((features, labels), open(cifar10_dataset_folder_path + '/' + save_as, 'wb'))

    _save_data('data_batch_1', 'preprocess_batch_1.p')
    _save_data('data_batch_2', 'preprocess_batch_2.p')
    _save_data('data_batch_3', 'preprocess_batch_3.p')
    _save_data('data_batch_4', 'preprocess_batch_4.p')
    _save_data('data_batch_5', 'preprocess_batch_5.p', end_index=0.5)
    _save_data('data_batch_5', 'preprocess_dev.p', start_index=0.5)
    _save_data('test_batch', 'preprocess_train.p')




def load_cfar10_batch_by_name(file):
    with open(file, mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return features, labels

def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    """
    Load a batch of the dataset
    """
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return features, labels

def display_stats(cifar10_dataset_folder_path, batch_id, sample_id):
    """
    Display Stats of the the dataset
    """
    batch_ids = list(range(1, 6))

    if batch_id not in batch_ids:
        print('Batch Id out of Range. Possible Batch Ids: {}'.format(batch_ids))
        return None

    features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_id)

    if not (0 <= sample_id < len(features)):
        print('{} samples in batch {}.  {} is out of range.'.format(len(features), batch_id, sample_id))
        return None

    print('\nStats of batch {}:'.format(batch_id))
    print('Samples: {}'.format(len(features)))
    print('Label Counts: {}'.format(dict(zip(*np.unique(labels, return_counts=True)))))
    print('First 20 Labels: {}'.format(labels[:20]))

    sample_image = features[sample_id]
    sample_label = labels[sample_id]
    label_names = _load_label_names()

    print('\nExample of Image {}:'.format(sample_id))
    print('Image - Min Value: {} Max Value: {}'.format(sample_image.min(), sample_image.max()))
    print('Image - Shape: {}'.format(sample_image.shape))
    print('Label - Label Id: {} Name: {}'.format(sample_label, label_names[sample_label]))
    plt.axis('off')
    plt.imshow(sample_image)
    plt.show()


def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]

import random
def batch_features_labels_triple(features1, features2, features3, batch_size):
    """
    Split features and labels into batches
    """
    random_features3 = []
    random_features2 = []

    for i in range(batch_size):
        random_features3.append(random.choice(features3))
        random_features2.append(random.choice(features2))

    for start in range(0, len(features1), batch_size):
        end = min(start + batch_size, len(features1))
        yield features1[start:end], np.array(random_features2), np.array(random_features3)



def load_preprocess_training_batch(batch_id, batch_size):
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    filename = 'cifar-10-batches-py/preprocess_batch_' + str(batch_id) + '.p'
    features, labels = pickle.load(open(filename, mode='rb'))

    # Return the training data in batches of size <batch_size> or less
    return batch_features_labels(features, labels, batch_size)


def _load_label_names():
    """
    Load the label names from file
    """
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

from skimage.transform import resize
def get_train_data():
    imgs = os.listdir('imgs')[:8000]
    #np.random.shuffle(X_train)
    data = []

    for img in imgs:
        real_image = ndimage.imread('imgs/' + img)
        # print(real_image.shape, resize(real_image,(300,400)).shape)
        # image = Image.fromarray(resize(real_image,(300,400)), 'RGB')
        # plt.imshow(image)
        # plt.show()
        image = Image.fromarray(real_image, 'RGB')
        image = image.resize((252,189))
        plt.imshow(image)
        plt.show()
        raise EOFError
        real_image = np.array(image)

        data.append(np.array(real_image) / 255)

    # raise EOFError
    return np.array(data)

def get_train_data2():
    X_train = []
    Y_train = []
    n = 300

    f = h5py.File(os.path.join('data', 'training_12-31-2017-0.hdf'), 'r')
    X_train_dataset = np.array(f.get('X_train'))
    Y_train_dataset = np.array(f.get('Y_train'))
    for i in range(n): X_train.append(X_train_dataset[i])
    for i in range(n): Y_train.append(Y_train_dataset[i])

    f = h5py.File(os.path.join('data', 'training_12-31-2017-1.hdf'), 'r')
    X_train_dataset = np.array(f.get('X_train'))
    Y_train_dataset = np.array(f.get('Y_train'))
    for i in range(n): X_train.append(X_train_dataset[i])
    for i in range(n): Y_train.append(Y_train_dataset[i])

    f = h5py.File(os.path.join('data', 'training_12-31-2017-2.hdf'), 'r')
    X_train_dataset = np.array(f.get('X_train'))
    Y_train_dataset = np.array(f.get('Y_train'))
    for i in range(n): X_train.append(X_train_dataset[i])
    for i in range(n): Y_train.append(Y_train_dataset[i])

    f = h5py.File(os.path.join('data', 'training_12-31-2017-3.hdf'), 'r')
    X_train_dataset = np.array(f.get('X_train'))
    Y_train_dataset = np.array(f.get('Y_train'))
    for i in range(n): X_train.append(X_train_dataset[i])
    for i in range(n): Y_train.append(Y_train_dataset[i])

    f = h5py.File(os.path.join('data', 'training_12-31-2017-4.hdf'), 'r')
    X_train_dataset = np.array(f.get('X_train'))
    Y_train_dataset = np.array(f.get('Y_train'))
    for i in range(n): X_train.append(X_train_dataset[i])
    for i in range(n): Y_train.append(Y_train_dataset[i])

    f = h5py.File(os.path.join('data', 'training_12-31-2017-5.hdf'), 'r')
    X_train_dataset = np.array(f.get('X_train'))
    Y_train_dataset = np.array(f.get('Y_train'))
    for i in range(n): X_train.append(X_train_dataset[i])
    for i in range(n): Y_train.append(Y_train_dataset[i])

    f = h5py.File(os.path.join('data', 'training_12-31-2017-6.hdf'), 'r')
    X_train_dataset = np.array(f.get('X_train'))
    Y_train_dataset = np.array(f.get('Y_train'))
    for i in range(n): X_train.append(X_train_dataset[i])
    for i in range(n): Y_train.append(Y_train_dataset[i])

    f = h5py.File(os.path.join('data', 'training_12-31-2017-7.hdf'), 'r')
    X_train_dataset = np.array(f.get('X_train'))
    Y_train_dataset = np.array(f.get('Y_train'))
    for i in range(n): X_train.append(X_train_dataset[i])
    for i in range(n): Y_train.append(Y_train_dataset[i])

    f = h5py.File(os.path.join('data', 'training_12-31-2017-8.hdf'), 'r')
    X_train_dataset = np.array(f.get('X_train'))
    Y_train_dataset = np.array(f.get('Y_train'))
    for i in range(n): X_train.append(X_train_dataset[i])
    for i in range(n): Y_train.append(Y_train_dataset[i])

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    np.random.shuffle(X_train), np.random.shuffle(Y_train)

    return X_train, Y_train



def ff_filter(images, labels):
    nx = []
    ny = []

    print('Filter data')

    for x, y in zip(images, labels):
        if abs(y) > 2 and abs(y) < 12:
            nx.append(x)
            ny.append(y)

    return np.array(nx), np.array(ny)


def triplet_loss(anchor, positive, negative, alpha = 0.2):
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)))
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)))
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # loss = basic_loss
    loss = tf.maximum(basic_loss, 0.0)
    return loss

from PIL import Image


def show_image(im):
    image = Image.fromarray(im, 'RGB')
    plt.imshow(image)
    plt.show()

# def triplet_loss(y_true, y_pred, alpha = 0.2):
#     """
#     Implementation of the triplet loss as defined by formula (3)
#
#     Arguments:
#     y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
#     y_pred -- python list containing three objects:
#             anchor -- the encodings for the anchor images, of shape (None, 128)
#             positive -- the encodings for the positive images, of shape (None, 128)
#             negative -- the encodings for the negative images, of shape (None, 128)
#
#     Returns:
#     loss -- real number, value of the loss
#     """
#
#     anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
#
#     ### START CODE HERE ### (≈ 4 lines)
#     # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
#     pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)))
#     # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
#     neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)))
#     # Step 3: subtract the two previous distances and add alpha.
#     basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
#     # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
#     loss = tf.maximum(basic_loss, 0.0)
#     ### END CODE HERE ###
#
#     return loss