import os
from typing import Callable, List, Tuple


import cv2


#
# インターフェースクラス定義
#


class IRecognizer:

    def recognize(self, img: cv2.Mat):
        pass


#
# インスタンス定義
#


# ここから下にクラスを追加する
