import os

import PIL
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt

def rgb2gray(rgb):
    filter = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]])
    return np.dot(rgb[..., :3], filter)
    # return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

image_path = '/home/srivoknovskiy/deepnets/gta-raod-examples/SenoraRd-GTAV-FuenteBlanca.jpg'
image_path = '/home/srivoknovskiy/deepnets/gta-raod-examples/9250dc48cf231b37db30d0723101d7e2370ed1f43c9fcd8dde32170825120048.jpg'
image_path = '/home/srivoknovskiy/deepnets/gta-raod-examples/e41e3547fc6a3e0edaadfc41767dd6f8459f5fcc6e7aef195bcf49269d058a28.jpg'

image = ndimage.imread(image_path)

# image = image.resize((120,90))
#image = image.convert(mode='P', dither=PIL.Image.FLOYDSTEINBERG, palette=PIL.Image.ADAPTIVE, colors=2)

for i in image:
    for j in i:
        # j[0] = j[0]*0.299
        # j[1] = j[1]*0.587
        # j[2] = j[2]*0.114
        less = 55
        if j[0] > less and j[1] > less and j[2] > less:
            j[2] = j[1] = j[0] = 0

# kernel = np.array([
#     [-5,5],
#     [-5,5],
# ])
# image = cv2.filter2D(image, -1, kernel)

# kernel = np.ones((3,3),np.float32)/9
# processed_image = cv2.filter2D(image,-1,kernel)

image = Image.fromarray(image, 'RGB')
plt.imshow(image)
plt.show()
#real_image = np.array(image)
