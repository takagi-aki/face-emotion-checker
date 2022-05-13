import os
from typing import List, Tuple
import time


import cv2
import dlib


#
# インターフェースクラス定義
#


class IDetector:

    def detect(self, img: cv2.Mat) -> List[Tuple[int, int, int, int]]:
        """Detect specified object in image.

        Args:
            img: CV2 BGR color image.

        Returns:
            List of Rectangle data.  
            Rectangle = [position-x, position-y, width, height].
        """
        pass


#
# インスタンス定義
#

class DetectorOpenCV(IDetector):
    def __init__(
        self, *,
        scaleFactor: float = 1.2,
        minNeighbors: int = 5,
        minSize: Tuple[int, int] = None,
        maxSize: Tuple[int, int] = None,
        **args
    ) -> None:
        data_dir = os.path.join(cv2.__path__[0], 'data')
        cascade_name = 'haarcascade_frontalface_default.xml'
        cascade_path = os.path.join(data_dir, cascade_name)
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.minSize = minSize
        self.maxSize = maxSize

    def detect(
        self,
        img: cv2.Mat
    ) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_position = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scaleFactor,
            minNeighbors=self.minNeighbors,
            minSize=self.minSize,
            maxSize=self.maxSize,
        )

        return faces_position


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

    CNNで顔を検出するので制度がよいが、恐ろしく遅いのであまり有効でない.

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


# ここから下にクラスを追加する
