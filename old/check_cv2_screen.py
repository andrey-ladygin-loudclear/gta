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

def preprocess_image(path):
    #img = cv2.imread(path, 0)

    image = ndimage.imread(path)
    image = Image.fromarray(image, 'RGB')
    #image = image.resize((252,189))
    image = image.resize((120,90))

    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow('window', image)
    time.sleep(0.2)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        #cv2.waitKey(0)
        cv2.destroyAllWindows()
        return


    real_image = np.array(image)
    return real_image

for dir in img_dir:
    for image in os.listdir(os.path.join(images_dir, dir)):
        images.append(preprocess_image(os.path.join(images_dir, dir, image)))
