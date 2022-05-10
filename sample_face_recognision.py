import glob
import os


import cv2


from fec.screen import Screen
from fec.camera import Camera
from fec.detector import DetectorOpenCV
from fec.recognizer.facenet import RecognizerFaceNet


face_dir = './image/face'

print('初期化中...')

sc = Screen()
camera = Camera()
ret, input_image = camera.frame()
face_detector = DetectorOpenCV()
face_recognizer = RecognizerFaceNet()

print('顔登録中...')
files = glob.glob(os.path.join(face_dir, '*.*'))
for file in files:
    name = os.path.splitext(os.path.basename(file))[0]

    img = cv2.imread(file)
    if(img is not None):
        Screen().show(img)
    
        face_positions = face_detector.detect(img)
        for (x, y, w, h) in face_positions:
            face_recognizer.register(name, img[y:y+h, x:x+w])

print('撮影開始')
ret, input_image = camera.frame()

while ret is True:
    face_positions = face_detector.detect(input_image)
    sc_img = input_image.copy()
    # clipping
    for (x, y, w, h) in face_positions:
        who, distance = face_recognizer.recognize(input_image[y:y+h,x:x+w])

        cv2.rectangle(sc_img, (x,y),(x+w,y+h), (0,0,255), 1)
        cv2.putText(sc_img, f'{who}={distance:.2f}', (x,y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

    sc.show(sc_img)


    k = cv2.waitKey(60) & 0xff
    if k == 27:
        break

    ret, input_image = camera.frame()
