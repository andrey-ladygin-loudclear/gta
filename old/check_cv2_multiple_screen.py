import cv2
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
import os
import time
import numpy as np

images = []
images_dir = '../imgs'
img_dir = os.listdir(images_dir)

def show_images2(iamges):
    #img = cv2.imread(path, 0)

    n = 0
    for path in iamges:
        image = ndimage.imread(path)
        image = Image.fromarray(image, 'RGB')
        #image = image.resize((252,189))
        image = image.resize((120,90))

        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imshow('window' + str(n), image)
        n += 1

    if cv2.waitKey(25) & 0xFF == ord('q'):
        #cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    time.sleep(5.2)


    real_image = np.array(image)
    return real_image

def show_images(iamges):
    im1 = images[0]
    im2 = images[2]

    image1 = ndimage.imread(im1)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = ndimage.imread(im2)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    numpy_horizontal = np.hstack((image1, image2))

    cv2.imshow('window', numpy_horizontal)
    cv2.waitKey(0)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        #cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    # time.sleep(5.2)

for dir in img_dir:
    for image in os.listdir(os.path.join(images_dir, dir)):
        images.append((os.path.join(images_dir, dir, image)))

show_images(images[0:4])




# import cv2
# import numpy as np
#
# image = cv2.imread('pinocchio.png')
# # I just resized the image to a quarter of its original size
# image = cv2.resize(image, (0, 0), None, .25, .25)
#
# grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# # Make the grey scale image have three channels
# grey_3_channel = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
#
# numpy_vertical = np.vstack((image, grey_3_channel))
# numpy_horizontal = np.hstack((image, grey_3_channel))
#
# numpy_vertical_concat = np.concatenate((image, grey_3_channel), axis=0)
# numpy_horizontal_concat = np.concatenate((image, grey_3_channel), axis=1)
#
# cv2.imshow('Main', image)
# cv2.imshow('Numpy Vertical', numpy_vertical)
# cv2.imshow('Numpy Horizontal', numpy_horizontal)
# cv2.imshow('Numpy Vertical Concat', numpy_vertical_concat)
# cv2.imshow('Numpy Horizontal Concat', numpy_horizontal_concat)
#
# cv2.waitKey()