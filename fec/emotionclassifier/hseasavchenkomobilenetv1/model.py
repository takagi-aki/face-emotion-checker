import tensorflow as tf
import onnxruntime as ort
import cv2
import numpy as np


from ..core import IEmotionClassifier

emo_list = ['Anger', 'Disgust', 'Fear',
            'Happiness',  'Neutral', 'Sadness', 'Surprise']


class EmotionClassifierHSEasavchenkoMobileNetv1(IEmotionClassifier):

    def __init__(self):
        file_path = './model/HSE-asavchenko.face-emotion-recognition/mobilenet_7.h5'
        self.model = tf.keras.models.load_model(file_path, compile=False)


    def classify(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224)).astype(dtype=np.float32)
        img = img.reshape((1, 224, 224, 3))
        return dict(zip(emo_list, self.model.predict(img)[0]))

class EmotionClassifierHSEasavchenkoMobileNetv1ONNX(IEmotionClassifier):

    def __init__(self):
        self.ort_sess = ort.InferenceSession('./model/HSE-asavchenko.face-emotion-recognition/model.onnx', providers=['CUDAExecutionProvider','CPUExecutionProvider'] )


    def classify(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224)).astype(dtype=np.float32)
        img = img.reshape((1, 224, 224, 3))
        predict = self.ort_sess.run(None, {'input_1':img})
        return dict(zip(emo_list, predict[0][0]))
