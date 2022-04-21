import imp
import tensorflow as tf
import cv2
import numpy as np


from ..core import IEmotionClassifier
from ...util.tf2 import load_frozen_graph

emo_list = ['Anger', 'Disgust', 'Fear',
            'Happiness',  'Neutral', 'Sadness', 'Surprise']


class EmotionClassifierMultitaskEfficientNetB2(IEmotionClassifier):

    def __init__(self):
        file_path = './model/HSE-asavchenko.face-emotion-recognition/mobilenet_7.h5'
        self.model = tf.keras.models.load_model(file_path, compile=False)


    def classify(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224)).astype(dtype=np.float32)
        img = img.reshape((1, 224, 224, 3))
        return [*zip(emo_list, self.model.predict(img)[0])]
