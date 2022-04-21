import glob
import os


import tensorflow as tf
import cv2
import numpy as np


from ..core import IRecognizer


class RecognizerFaceNet(IRecognizer):

    def __init__(self):
        filename = './model/FaceNet/20180402-114759.pb'

        with tf.io.gfile.GFile(filename, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            loaded = graph_def.ParseFromString(f.read())

        def _imports_graph_def():
            tf.compat.v1.import_graph_def(graph_def, name="")

        func_imported = tf.compat.v1.wrap_function(_imports_graph_def, [])
        graph_imported = func_imported.graph

        self.frozen_func = func_imported.prune(
            tf.nest.map_structure(
                graph_imported.as_graph_element,
                ['input:0', 'phase_train:0']),
            tf.nest.map_structure(
                graph_imported.as_graph_element, 'embeddings:0')
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
        resized_img = cv2.resize(img, (160, 160))
        input_img = (resized_img.reshape(
            (1, 160, 160, 3)).astype(np.float32) / 255)

        emb = self.frozen_func(
            input=tf.constant(self._prewhiten(input_img), tf.float32),
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
