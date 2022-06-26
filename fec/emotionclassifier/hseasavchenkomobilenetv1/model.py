from math import atan2, pi

import onnxruntime as ort
import cv2
import numpy as np

from ..core import IEmotionClassifier

emo_list = ['Anger', 'Disgust', 'Fear',
            'Happiness',  'Neutral', 'Sadness', 'Surprise']


def _face_cliping(img, face):
    x,y,w,h = map(int, face[0:4])
    rx,ry,lx,ly = face[4:8]

    # 顔部分の切り抜き
    img = img[y:y+h,x:x+w]

    # 目線が平行になるように回転
    rotate_mat = cv2.getRotationMatrix2D((w/2,y/2), atan2(ly-ry,lx-rx) * 180 / pi, 1)
    img = cv2.warpAffine(img, rotate_mat, (w,h))

    # フォーマットを調整 float32 RGB 244x244x3
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224)).astype(dtype=np.float32)
    img = img.reshape((1, 224, 224, 3))
    return img


class EmotionClassifierHSEasavchenkoMobileNetv1(IEmotionClassifier):

    def __init__(self):
        import tensorflow as tf
        file_path = './model/HSE-asavchenko.face-emotion-recognition/mobilenet_7.h5'
        self.model = tf.keras.models.load_model(file_path, compile=False)


    def classify(self, img, face):
        img = _face_cliping(img, face)

        return dict(zip(emo_list, self.model.predict(img)[0]))

class EmotionClassifierHSEasavchenkoMobileNetv1ONNX(IEmotionClassifier):

    def __init__(self):
        self.ort_sess = ort.InferenceSession('./model/HSE-asavchenko.face-emotion-recognition/model.onnx', providers=['CUDAExecutionProvider','CPUExecutionProvider'] )


    def classify(self, img, face):
        img = _face_cliping(img, face)

        predict = self.ort_sess.run(None, {'input_1':img})
        return dict(zip(emo_list, predict[0][0]))
