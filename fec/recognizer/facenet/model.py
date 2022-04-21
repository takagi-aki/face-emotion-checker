import imp
import tensorflow as tf
import cv2
import numpy as np


from ..core import IRecognizer
from ...util.tf2 import load_frozen_graph

class RecognizerFaceNet(IRecognizer):

    def __init__(self):
        file_path = './model/FaceNet/20180402-114759.pb'
        input_layer_names = ['input:0', 'phase_train:0']
        output_layer_names = 'embeddings:0'

        self.frozen_func = load_frozen_graph(
            file_path,
            input_layer_names,
            output_layer_names
        )

        self.saved_face_vec = dict()

    def register(self, name, img):
        emb = self.embedding(img)
        self.saved_face_vec[name] = emb

    def _calc_euclid_distance(self, a, b):
        return np.linalg.norm(a-b)

    def _prewhiten(self, x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1/std_adj)
        return y

    def embedding(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (160, 160)).astype(np.float32)
        img = self._prewhiten(img)
        img = img.reshape((1,160,160,3))

        emb = self.frozen_func(
            input=tf.constant(img, tf.float32),
            phase_train=tf.constant(False, tf.bool))
        return emb

    def recognize(self, img):
        emb = self.embedding(img)

        most_similar = '?'
        distance_similar = 1000
        for name, vec in self.saved_face_vec.items():
            distance = self._calc_euclid_distance(emb, vec)
            if distance_similar > distance:
                most_similar = name
                distance_similar = distance
        return most_similar, distance_similar
