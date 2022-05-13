import os
from typing import List, Tuple


import cv2
import dlib
import numpy as np

#
# インターフェースクラス定義
#


class ILandmarkDetector:

    def detect(self, img: cv2.Mat) -> List[Tuple[int, int]]:
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

class LandmarkDetectorDlib(ILandmarkDetector):
    def __init__(
        self,
        **args
    ) -> None:
        self.predictor = dlib.shape_predictor("model\dlib\shape_predictor_68_face_landmarks.dat")


    def detect(
        self,
        img: cv2.Mat,
        faces
    ):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x,y,w,h = faces
        rect = dlib.rectangle(x,y,x+w,y+h)
        detection = self.predictor(img, rect)
        buffer = np.zeros(shape=(68, 2), dtype=np.integer)
        for i in range(68):
            x = detection.part(i).x
            y = detection.part(i).y
            buffer[i] = (int(x),int(y))
        return buffer


# ここから下にクラスを追加する
