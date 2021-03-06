import cv2
import numpy as np
import time
import RPi.GPIO as GP

GP.setmode(GP.BOARD)
GP.setwarnings(False)

thres = 0.45 # Threshold to detect object
nmsthres = 0.1

#/home/danieletostiPI/ROBOT_FACE_15/
classNames= []
classFile = '/home/danieletostiPI/ROBOT_FACE_15/coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = '/home/danieletostiPI/ROBOT_FACE_15/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = '/home/danieletostiPI/ROBOT_FACE_15/frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

#----------------------------------------------------------------------------------------------
LEDG = 16
BUTTON_STOP = 18

SERVO_ROT = 32

GP.setup(LEDG, GP.OUT)
GP.setup(SERVO_ROT, GP.OUT)

GP.setup(BUTTON_STOP, GP.IN, pull_up_down=GP.PUD_UP)  # Se Button = 0, bottone pigiato.

servo_rot = GP.PWM(SERVO_ROT, 50)  # 50 Hz

#servo_rot.ChangeDutyCycle(0)  # fermo # va da 2.2 a 12
#servo_lat.ChangeDutyCycle(0)

DutyCycle1 = 27
dc1 = float(DutyCycle1)

def rotate_right(dc1, inc):
    if dc1 <= 9:
        dc1 = 9
    else:
        dc1 = dc1 - inc
    print("ROT RIGHT")
    servo_rot.ChangeDutyCycle(dc1 / 4)
    time.sleep(0.05)
    servo_rot.ChangeDutyCycle(0)
    return dc1


def rotate_left(dc1, inc):
    if dc1 >= 42:
        dc1 = 42
    else:
        dc1 = dc1 + inc
    print("ROT LEFT")
    servo_rot.ChangeDutyCycle(dc1 / 4)
    time.sleep(0.05)
    servo_rot.ChangeDutyCycle(0)
    return dc1
#----------------------------------------------------------------------------------------------

def getObjects(img ,thres, nmsthres, objects = []):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nmsthres)
    #print(classIds,bbox)
    ObjectInfo = []

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                ObjectInfo.append([box])
                cv2.rectangle(img,box,color=(0,0,255),thickness=2)
    return img, ObjectInfo

#----------------------------------------------------------------------------------------------
go = True
stop = True
spin = 0
enablel = 1
enabler = 1

xres = 320
yres = 180

servo_rot.start(6.8)
#----------------------------------------------------------------------------------------------

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, xres)
    cap.set(4, yres)
    print("Paletto 1")
    while go:
        servo_rot.ChangeDutyCycle(0)
        # --------------------------
        stop = GP.input(BUTTON_STOP)
        GP.output(LEDG, GP.HIGH)
        if not stop:
            go = False
            GP.output(LEDG, GP.LOW)
            print("Well Done")
        # --------------------------
        try:
            success, img = cap.read()
            result, box_array = getObjects(img, thres, nmsthres, objects = ['person'])
            #print(box_array)  # x, y (angolo alto sinistra), width,height
            if len(box_array) != 0:
                x = box_array[0][0][0]
                y = box_array[0][0][1]
                w = box_array[0][0][2]
                h = box_array[0][0][3]
            else:
                x = 0
                y = 0
                h = 0
                w = 0
            xcenter = x + w / 2
            ycenter = y + h / 2

        except:
            xcenter = 0
            ycenter = 0
            print("Empty")

        #print("Xcenter = ", xcenter)
        #print("Ycenter = ", ycenter)

        if xcenter == 0 and ycenter == 0 and spin == 0:
            print("Spin")
            spin = 1
        elif xcenter != 0 and ycenter != 0 and spin == 1:
            print("Stop motor and continue")
            spin = 0
        if xcenter < xres / 2 - 30 and spin == 0:
            # print("gira a destra")
            enablel = 1
            if enabler == 1:
                if xcenter < xres / 2 - 110:
                    inc = 6
                elif xcenter < xres / 2 - 90:
                    inc = 4
                elif xcenter < xres / 2 - 60:
                    inc = 2
                else:
                    inc = 1
                dc1 = rotate_left(dc1, inc)
                time.sleep(0.005)
                #print("Increase right :", inc)
            if dc1 <= 9:
                enabler = 0

        elif xcenter > xres / 2 + 30 and spin == 0:
            # print("gira a sinistra")
            enabler = 1
            if enablel == 1:
                if xcenter > xres / 2 + 110:
                    inc = 6
                elif xcenter > xres / 2 + 90:
                    inc = 4
                elif xcenter > xres / 2 + 60:
                    inc = 2
                else:
                    inc = 1
                dc1 = rotate_right(dc1, inc)
                time.sleep(0.005)
                #print("Increase left :", inc)
            if dc1 >= 42:
                enablel = 0
        #cv2.imshow("Output", img)
        cv2.waitKey(1)