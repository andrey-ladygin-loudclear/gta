import gzip
from threading import Thread

import tables
import numpy as np
import cv2
import time
import os

import zlib
from PIL import ImageGrab
from getkeys import key_check
from player import GTAPlayer
import datetime
import pickle

player = GTAPlayer()
start = False

X_train = []
Y_train = []


def grab():
    global start
    last_time = time.time()
    number_of_image = 0
    time_sum = 0

    while(True):
        check_commands()

        if(start):
            screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
            image = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            cv2.imshow('window', image)

            if number_of_image % 100 == 0:
                mean_time_diff = time_sum / 100
                time_sum = 0
                print('Loop took {}, shape is {}'.format(mean_time_diff, image.shape))
            time_sum += time.time() - last_time

            folder = number_of_image // 1000
            if not os.path.isdir("imgs/"+str(folder)):
                os.makedirs("imgs/"+str(folder))
            cv2.imwrite("imgs/"+str(folder)+"/"+str(number_of_image)+".jpg", image)
            number_of_image += 1

            addToData(image)

            if len(Y_train) == 1000:
                print('SAVE', len(Y_train), number_of_image)
                saveTrainingData()

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

            last_time = time.time()

isFirstCheckCommands = True
def check_commands():
    global start, isFirstCheckCommands, X_train, Y_train
    pressed_keys = key_check()

    if isFirstCheckCommands:
        isFirstCheckCommands = False
        return

    if 'CTRL' in pressed_keys:
        if 'T' in pressed_keys:
            start = True
        if 'S' in pressed_keys:
            saveTrainingData()
        if 'E' in pressed_keys:
            start = False
        if 'C' in pressed_keys:
            X_train = []
            Y_train = []


n = 0
def addToData(image):
    global Y_train, n
    y = [1, 0, 0]
    pressed_keys = key_check()

    # if 'A' in pressed_keys: y[0] = 1
    # if 'W' in pressed_keys: y[1] = 1
    # if 'D' in pressed_keys: y[2] = 1
    # if 'S' in pressed_keys: y[3] = 1

    if 'A' in pressed_keys: y = [0, 1, 0]
    if 'S' in pressed_keys: y = [0, 0, 1]

    #image = image / 255
    # training_data.append({
    #     'x': image,
    #     'y': y
    # })
    #X_train.append(image)
    Y_train.append(y)

    # if len(Y_train) >= 1000:
    #training_data.append(image)

def getFileName(date, iteration):
    return 'data/training_{}-{}.hdf'.format(date, iteration)

i = 0
def saveTrainingData():
    global X_train, Y_train, i
    if len(Y_train) > 0:
        print('SaveTrainingData: ', len(Y_train))
        np.save('labels/'+str(i) + '-labels', Y_train)
        Y_train = []
        i += 1
    return

    date = datetime.datetime.now()
    date = date.strftime('%m-%d-%Y')
    file_name = getFileName(date, i)

    while os.path.isfile(file_name):
        i += 1
        file_name = getFileName(date, i)

    print(file_name)

    thread = Thread(target=save_file, args=(file_name, X_train, Y_train))
    thread.setDaemon(True)
    thread.start()
    time.sleep(0.5)

    i += 1
    X_train = []
    Y_train = []

Cobj = zlib.compressobj(level=-1)

def save_file(file_name, X_train, Y_train):
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    print('saving training data', len(X_train))

    # data = zlib.compress(np.array(training_data).tobytes())
    # np.save(file_name+'.c', data)
    # np.save(file_name+'.b', np.array(training_data).tobytes())
    # np.save(file_name, X_train)

    f = tables.openFile(file_name, 'w')
    filters = tables.Filters(complib='blosc', complevel=5)

    atom1= tables.Atom.from_dtype(X_train.dtype)
    ds1 = f.createCArray(f.root, 'X_train', atom1, X_train.shape, filters=filters)
    ds1[:] = X_train

    atom2 = tables.Atom.from_dtype(X_train.dtype)
    ds2 = f.createCArray(f.root, 'Y_train', atom2, Y_train.shape, filters=filters)
    ds2[:] = Y_train
    f.close()

    # import gzip
    # content = "Lots of content here"
    # with gzip.open('file.txt.gz', 'wb') as f:
    #     f.write(content)

    print('Training Data saved!')


# screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
# new_screen = process_img(screen)
# #printscreen_numpy = np.array(printscreen_pil.getdata(),dtype='uint8')
# #printscreen_numpy = np.array(printscreen_pil.getdata(),dtype='uint8').reshape((printscreen_pil.size[1],printscreen_pil.size[0],3))
#
# # cv2.imshow('window', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
# cv2.imshow('window', new_screen)
# #cv2.imwrite("imgs/im-" + str(i) + ".jpg", printscreen_numpy)
# print('Loop took {}'.format(time.time() - last_time))
# last_time = time.time()
# i+=1
#
# if cv2.waitKey(25) & 0xFF == ord('q'):
#     cv2.destroyAllWindows()
#     break