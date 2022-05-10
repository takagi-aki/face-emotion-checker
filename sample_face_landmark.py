import glob
import os


import cv2


from fec.screen import Screen
from fec.camera import Camera
from fec.detector import DetectorOpenCV
from fec.landmarkdetector import LandmarkDetectorOpenCV


face_dir = './image/face'

print('初期化中...')

sc = Screen()
camera = Camera()
ret, input_image = camera.frame()
face_detector = DetectorOpenCV()
face_landmarkd = LandmarkDetectorOpenCV()


print('撮影開始')
ret, input_image = camera.frame()

while ret is True:
    face_positions = face_detector.detect(input_image)
    sc_img = input_image.copy()
    # clipping
    for face in face_positions:
        a = face_landmarkd.detect(input_image, face)

        #print(a.parts())

        for x, y in a:
            cv2.circle(sc_img, (x,y), 2, (0,0,255), 2)

    sc.show(sc_img)


    k = cv2.waitKey(60) & 0xff
    if k == 27:
        break

    ret, input_image = camera.frame()
