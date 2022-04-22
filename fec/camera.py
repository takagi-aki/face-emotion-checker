from typing import Tuple


import cv2


class Camera:

    def __init__(self, size=(720, 480)) -> None:
        self.cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])

    @property
    def is_enable(self) -> bool:
        return self.cam.isOpened()

    def frame(self) -> Tuple[bool, cv2.Mat]:
        ret, img = self.cam.read()
        return ret, img
