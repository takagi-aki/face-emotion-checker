import os
from typing import Callable, List, Tuple


import numpy as np
import cv2


#
# インターフェースクラス定義
#


class IDetector:

    def detect(self, img: List[Tuple[int, int, int, int]]):
        pass


#
# 関数で使うデコレータ定義
#


_detector_dict: dict[str, Callable[[], IDetector]] = dict()


def _detector_cls(name):
    def deco(Cls):
        _detector_dict[name] = Cls
    return deco


#
# ファクトリー定義
#


def get_detector(name: str) -> IDetector:
    dcls = _detector_dict.get(name)
    if(dcls is None):
        raise ValueError
    return dcls()


#
# インスタンス定義
#


@_detector_cls(name='OpenCV')
class _DetectorOpenCV(IDetector):
    def __init__(self) -> None:
        data_dir = os.path.join(cv2.__path__[0], 'data')
        cascade_name = 'haarcascade_frontalface_default.xml'
        cascade_path = os.path.join(data_dir, cascade_name)
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, img: cv2.Mat) -> List[Tuple[int, int, int, int]]:

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_position = self.face_cascade.detectMultiScale(
            gray, 1.3, 5)

        return faces_position


# ここから下にクラスを追加する
