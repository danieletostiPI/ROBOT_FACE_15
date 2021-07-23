import cv2
import numpy as np
import time
import RPi.GPIO as GP

GP.setmode(GP.BOARD)
GP.setwarnings(False)

thres = 0.45 # Threshold to detect object
nmsthres = 0.2

#/home/danieletostiPI/ROBOT_FACE_15/
classNames= []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

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

#servo_rot.ChangeDutyCycle(0)  # fermo # va da 2.2 a 12.25
#servo_lat.ChangeDutyCycle(0)

DutyCycle1 = 27
dc1 = float(DutyCycle1)

def rotate_right(dc1, inc):
    if dc1 <= 9:
        dc1 = 9
    else:
        dc1 = dc1 - inc
    servo_rot.ChangeDutyCycle(dc1 / 4)
    time.sleep(0.02)
    servo_rot.ChangeDutyCycle(0)
    return dc1


def rotate_left(dc1, inc):
    if dc1 >= 42:
        dc1 = 42
    else:
        dc1 = dc1 + inc
    servo_rot.ChangeDutyCycle(dc1 / 4)
    time.sleep(0.02)
    servo_rot.ChangeDutyCycle(0)
    return dc1
#----------------------------------------------------------------------------------------------

def getObjects(img ,thres, nmsthres, objects = []):

    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nmsthres)
    #print(classIds,bbox)
    ObjectInfo = []

    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            if className in objects:
                ObjectInfo.append([box,className])
                cv2.rectangle(img,box,color=(0,0,255),thickness=2)
    return img, ObjectInfo

#----------------------------------------------------------------------------------------------
go = True
right = True
left = True
stop = True

spin = 0
inc = 1
enabler = 1
enablel = 1

servo_rot.start(6.8)
#----------------------------------------------------------------------------------------------

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    while go:
        # --------------------------
        stop = GP.input(BUTTON_STOP)
        GP.output(LEDG, GP.HIGH)
        if not stop:
            go = False
            GP.output(LEDG, GP.LOW)
            print("Well Done")
        # --------------------------
        success, img = cap.read()
        result, ObjectInfo = getObjects(img, thres, nmsthres, objects = ['person'])
        print(ObjectInfo)
        cv2.imshow("Output", img)
        cv2.waitKey(1)