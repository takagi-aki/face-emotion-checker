import glob
import os

import cv2


from fec.screen import Screen
from fec.camera import Camera
from fec.detector import DetectorOpenCV
from fec.emotionclassifier.hseasavchenkomobilenetv1 import EmotionClassifierHSEasavchenkoMobileNetv1


face_dir = './image/face'

print('初期化中...')

sc = Screen()
camera = Camera()
ret, input_image = camera.frame()
face_detector = DetectorOpenCV()
face_classifier = EmotionClassifierHSEasavchenkoMobileNetv1()


print('撮影開始')
ret, input_image = camera.frame()

while ret is True:
    face_positions = face_detector.detect(input_image)
    sc_img = input_image.copy()
    # clipping
    for (x, y, w, h) in face_positions:
        ret = face_classifier.classify(input_image[y:y+h,x:x+w])
        emotion, probability = max(ret, key= lambda x : x[1])

        cv2.rectangle(sc_img, (x,y),(x+w,y+h), (0,0,255), 1)
        cv2.putText(sc_img, f'{emotion}={probability:.2f}', (x,y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

    sc.show(sc_img)


    k = cv2.waitKey(60) & 0xff
    if k == 27:
        break

    ret, input_image = camera.frame()
