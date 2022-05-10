from typing import Dict


import cv2


#
# インターフェースクラス定義
#


class IEmotionClassifier:

    def classify(self, img: cv2.Mat) -> Dict[str, float]:
        """Classify emotion of face image.

        Args:
            img: CV2 BGR color image.

        Returns:
            Key is EmothionTag(ex. Happiness). Value is Possibility.
        """
        pass
