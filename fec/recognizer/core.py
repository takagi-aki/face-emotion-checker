from typing import Tuple


import cv2


#
# インターフェースクラス定義
#


class IRecognizer:

    def recognize(self, img: cv2.Mat, face) -> Tuple[str, float]:
        """Gain name and distance bitween registered before from face image.

        Args:
            img: CV2 BGR color image.
        
        Returns:
            Return Tuple. First is name of face, Second is distance.
        """
        pass


#
# インスタンス定義
#


# ここから下にクラスを追加する
