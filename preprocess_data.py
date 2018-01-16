import os
import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt

# a = np.array([1,2,3,4,5,6,7,8,9])
# b = np.array([11,22,33,44,55,66,77,88,99])
#
# s = np.arange(a.shape[0])
# np.random.shuffle(s)
#
# print(s)
# print(a[s])
# print(b[s])
#
# raise ValueError

images_dir = 'imgs'
labels_dir = 'labels'

img_dir = os.listdir(images_dir)
lab_dir = os.listdir(labels_dir)

images = []
labels = []

def preprocess_image(image_path):
    real_image = ndimage.imread(image_path)
    image = Image.fromarray(real_image, 'RGB')
    #image = image.resize((252,189))
    image = image.resize((120,90))
    # plt.imshow(image)
    # plt.show()
    real_image = np.array(image)
    return real_image

def preprocess_label(np_lables):
    return np_lables

def filter_data(images, labels):
    nx = []
    ny = []

    print('Filter data')

    for x, y in zip(images, labels):
        diff = y[0] - y[2]
        if diff:
            nx.append(x)
            ny.append([diff])

    return np.array(nx), np.array(ny)

def straight_road_filter(images, labels):
    nx = []
    ny = []

    print('Straight Road Filter data')

    for x, y in zip(images, labels):
        diff = y[0] - y[2]
        if not diff:
            nx.append(x)
            ny.append([y[1]])

    return np.array(nx), np.array(ny)

def one_hot_encode(x, m):
    n = len(x)
    b = np.zeros((n, m))
    b[np.arange(n), x] = 1
    return b
# def one_hot_encode(x):
#     n = len(x)
#     b = np.zeros((n, max(x)+1))
#     b[np.arange(n), x] = 1
#     return b

print('Processing Images')
for dir in img_dir:
    for image in os.listdir(os.path.join(images_dir, dir)):
        images.append(preprocess_image(os.path.join(images_dir, dir, image)))

print("Processing Labels")
for np_lables in lab_dir:
    numpy_data = np.load(os.path.join(labels_dir, np_lables))
    for data in numpy_data:
        labels.append(data)


def save_data(x, y, shuffle=True, x_name='features', y_name='labels'):
    if shuffle:
        s = np.arange(x.shape[0])
        np.random.shuffle(s)
        x = x[s]
        y = y[s]

    print('save: ', x_name, x.shape)
    print('save: ', y_name, y.shape)

    np.save(os.path.join('data', x_name), x)
    np.save(os.path.join('data', y_name), y)


images = np.array(images)
labels = np.array(labels)

print('old: ', images.shape)
print('old: ', labels.shape)

x, y = filter_data(images, labels)
save_data(x, y, x_name='features', y_name='labels')

x, y = straight_road_filter(images, labels)
save_data(x, y, x_name='straights_road_features', y_name='straights_road_labels')
