import cv2
import numpy as np
import time
import PoseModule as pm

cap = cv2.VideoCapture("PoseVideos/pushUp42.mp4")

detector = pm.poseDetector()

while True:
    success, img = cap.read()
    img = cv2.resize(img, (1288, 720))
    #img = cv2.imread("PoseVideos/Plank1.png")
    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)
    #print(lmList)
    if len(lmList) != 0:
        #Right Elbow
        #detector.findAngle(img, 12,14,16)
        #Left Elbow
        detector.findAngle(img, 11,13,15)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

