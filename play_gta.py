import numpy as np
import cv2
import time
from PIL import ImageGrab
from getkeys import key_check
from prediction_road_turn import predict
from player import GTAPlayer

player = GTAPlayer()
start = False


def grab():
    global start
    last_time = time.time()
    number_of_image = 0
    time_sum = 0

    while(True):
        check_commands()

        if(start):
            screen = ImageGrab.grab(bbox=(0, 40, 800, 640))
            screen = screen.resize((120,90))
            screen = np.array(screen)
            image = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            cv2.imshow('window', image)

            if number_of_image % 100 == 0:
                mean_time_diff = time_sum / 100
                time_sum = 0
                print('Loop took {}, shape is {}'.format(mean_time_diff, image.shape))
            time_sum += time.time() - last_time

            get_prediction(image / 255)

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
        if 'E' in pressed_keys:
            start = False


def get_prediction(image):
    prediction = predict([image])
    turnLeft = True

    if prediction[0] < 0:
        turnLeft = False



    player.forward()
    time.sleep(0.3)
    player.releaseForward()

    if turnLeft:
        player.left()

    if not turnLeft:
        player.right()

    time.sleep(abs(prediction[0]) / 250)
    player.releaseLeft()
    player.releaseRight()

grab()