from typing import Tuple


import cv2


class Camera:

    def __init__(self) -> None:
        self.cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    @property
    def is_enable(self) -> bool:
        return self.cam.isOpened()

    def frame(self) -> Tuple[bool, cv2.Mat]:
        ret, img = self.cam.read()
        return ret, img
