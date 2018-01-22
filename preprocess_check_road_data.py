import os
import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt


img_dir = ['D:/datasets/non-road/1', 'D:/datasets/non-road/2', 'D:/datasets/non-road/3', 'D:/datasets/non-road/19', 'D:/datasets/non-road/20']

images = []

def preprocess_image(image_path):
    real_image = ndimage.imread(image_path)
    image = Image.fromarray(real_image, 'RGB')
    image = image.resize((120,90))
    real_image = np.array(image)
    return real_image

for dir in img_dir:
    for image in os.listdir(dir):
        images.append(preprocess_image(os.path.join(dir, image)))


def save_data(x, shuffle=True, x_name='features'):
    if shuffle:
        s = np.arange(x.shape[0])
        np.random.shuffle(s)
        x = x[s]

    print('save: ', x_name, x.shape)

    np.save(os.path.join('data', x_name), x)


images = np.array(images)

print('old: ', images.shape)

save_data(images,  x_name='non_road_features2')
