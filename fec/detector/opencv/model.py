from typing import List, Tuple


import cv2


from ..core import IDetector


class DetectorOpenCVYN(IDetector):
    """CV2とYuNetで顔検出.

    CV2標準のhaarcascade判別機と変わらず高速で、さらにそれより精度が高い.
    対象の画素数が大きすぎると判定できないことがある.

    モデルを以下のリンクからダウンロードする必要あり.
    https://github.com/ShiqiYu/libfacedetection.train/tree/master/tasks/task1/onnx
    """
    def __init__(
        self,
        **args
    ) -> None:
        self.detector  = cv2.FaceDetectorYN_create("model/opencv/yunet.onnx", "", (0, 0))

    def detect(
        self,
        img: cv2.Mat
    ) -> List[Tuple[int, int, int, int]]:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.detector.setInputSize((img.shape[1], img.shape[0]))
        _, faces = self.detector.detect(img)
        if faces is None:
            faces = []

        faces_position = []

        return faces

