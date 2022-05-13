from typing import List, Tuple


import cv2
from mtcnn.mtcnn import MTCNN


from ..core import IDetector


class DetectorMTCNN(IDetector):
    """MTCNNで顔検出.

    cv2やdlibに比べ若干重い.性能は非常に良い.
    """
    def __init__(
        self,
        **args
    ) -> None:
        self.detector = MTCNN()

    def detect(
        self,
        img: cv2.Mat
    ) -> List[Tuple[int, int, int, int]]:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faces = self.detector.detect_faces(img)

        faces_position = []
        for face in faces:
            faces_position.append(face['box'])

        return faces_position

