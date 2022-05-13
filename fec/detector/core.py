import os
from typing import List, Tuple


import cv2


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

# ここから下にクラスを追加する
