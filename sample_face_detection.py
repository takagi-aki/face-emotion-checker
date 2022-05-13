import time
import glob
import os

import cv2


from fec.screen import Screen
from fec.camera import Camera
from fec.detector import DetectorDibCNN as Ditector


print('初期化中...')

sc = Screen()
camera = Camera()
ret, input_image = camera.frame()
face_detector = Ditector()


print('撮影開始')
ret, input_image = camera.frame()

while ret is True:
    face_positions = face_detector.detect(input_image)
    sc_img = input_image.copy()
    for (x, y, w, h) in face_positions:
        cv2.rectangle(sc_img, (x,y),(x+w,y+h), (0,0,255), 1)

    sc.show(sc_img)

    k = cv2.waitKey(60) & 0xff
    if k == 27:
        break

    ret, input_image = camera.frame()
