import time
import glob
import os

import cv2


from screen import Screen
from camera import Camera
from detector import DetectorOpenCV

face_dir = './image/face'

print('初期化中...')

sc = Screen()
face_detector = DetectorOpenCV()

print('顔認識開始...')
files = glob.glob(os.path.join(face_dir, '*.*'))
for file in files:
    name = os.path.splitext(os.path.basename(file))

    img = cv2.imread(file)
    sc.show(img)
    while True:
        time.sleep(0.1)
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break

    face_positions = face_detector.detect(img)
    for (x, y, w, h) in face_positions:
        cv2.rectangle(img, (x,y),(x+w,y+h), (255,0,0), 3)
    
    sc.show(img)

while True:
    time.sleep(0.1)
    k = cv2.waitKey(60) & 0xff
    if k == 27:
        break



