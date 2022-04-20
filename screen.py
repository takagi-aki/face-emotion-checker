import time


import cv2


class Screen:

    def __init__(self, name: str = None) -> None:
        if(name is not None):
            self.name = name
        else:
            self.name = str(time.time())

    def show(self, img: cv2.Mat):
        cv2.imshow(self.name, img)
