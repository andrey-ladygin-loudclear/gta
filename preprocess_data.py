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

images_dir = '/home/srivoknovskiy/deepnets/data/imgs'
labels_dir = '/home/srivoknovskiy/deepnets/data/labels'

img_dir = os.listdir(images_dir)
lab_dir = os.listdir(labels_dir)

images = []
labels = []

def preprocess_image(image_path):
    real_image = ndimage.imread(image_path)
    image = Image.fromarray(real_image, 'RGB')
    image = image.resize((252,189))
    # plt.imshow(image)
    # plt.show()
    real_image = np.array(image)
    return real_image

def preprocess_label(np_lables):
    return np_lables

def filter_data(images, labels):
    nx = []
    ny = []

    print('old: ', images.shape)
    print('old: ', labels.shape)

    for x, y in zip(images, labels):
        if y[0] != 0:
            nx.append(x)
            ny.append([1, 0, 0])
        if y[2] != 0:
            nx.append(x)
            ny.append([0, 1, 0])
        if y[0] == 0 and y[2] == 0:
            nx.append(x)
            ny.append([0, 0, 1])

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


images = np.array(images)
labels = np.array(labels)

s = np.arange(images.shape[0])
np.random.shuffle(s)

images = images[s]
labels = labels[s]

print('labels sum: ', sum(labels))

#images, labels = filter_data(images, labels)

print('save filtered images: ', images.shape)
print('save filtered labels: ', labels.shape)

np.save('data/features', images)
np.save('data/labels', labels)