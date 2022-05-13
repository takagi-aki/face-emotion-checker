from typing import List, Tuple


import cv2
import dlib


from ..core import IDetector


class DetectorDib(IDetector):
    def __init__(
        self,
        **args
    ) -> None:
        self.detector = dlib.get_frontal_face_detector()

    def detect(
        self,
        img: cv2.Mat
    ) -> List[Tuple[int, int, int, int]]:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = self.detector(img, 1)
        faces_position = []
        for i, d in enumerate(dets):
            faces_position.append((d.left(), d.top(), d.width(), d.height()))

        return faces_position


class DetectorDibCNN(IDetector):
    """Dibのcnn_face_detectorで顔検出.

    CNNで顔を検出するので精度がよいが、恐ろしく遅い.

    """

    def __init__(
        self,
        **args
    ) -> None:
        self.detector = dlib.cnn_face_detection_model_v1(
            './model/dlib/mmod_human_face_detector.dat')

    def detect(
        self,
        img: cv2.Mat
    ) -> List[Tuple[int, int, int, int]]:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dets = self.detector(img, 1)
        faces_position = []
        for i, d in enumerate(dets):
            faces_position.append(
                (d.rect.left(), d.rect.top(), d.rect.width(), d.rect.height()))

        return faces_position