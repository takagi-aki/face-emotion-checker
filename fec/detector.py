import os
from typing import Callable, List, Tuple


import cv2


#
# インターフェースクラス定義
#


class IDetector:

    def detect(self, img: List[Tuple[int, int, int, int]]):
        pass


#
# インスタンス定義
#

class DetectorOpenCV(IDetector):
    def __init__(self) -> None:
        data_dir = os.path.join(cv2.__path__[0], 'data')
        cascade_name = 'haarcascade_frontalface_default.xml'
        cascade_path = os.path.join(data_dir, cascade_name)
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, img: cv2.Mat) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_position = self.face_cascade.detectMultiScale(
            gray, 1.2, 5)

        return faces_position


# ここから下にクラスを追加する
