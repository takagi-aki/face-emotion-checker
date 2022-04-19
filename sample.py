from camera import Camera
from detector import get_detector
from recognizer import get_recognizer
from emotionclassifier import get_classifier

camera = Camera()
face_detector = get_detector('backend_name')
face_recognizer = get_recognizer('backend_name')
face_classifier = get_classifier('backend_name')


input_image = camera.frame()

faces_images = face_detector.detect(input_image)
for face_img in faces_images:
    who = face_recognizer.recognize(face_img)
    emo = face_classifier.classify(face_img)

    print(who, emo)
