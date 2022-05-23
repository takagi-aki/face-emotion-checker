from re import X
import cv2
import numpy as np


def clip(img: cv2.Mat, x, y, w, h):

    result = np.zeros((h, w, img.shape[2]), dtype=np.uint8)
    cp_l = max(0, x)
    cp_t = max(0, y)
    cp_r = min(img.shape[1], x + w)
    cp_b = min(img.shape[0], y + h)

    result[cp_t-y: cp_b-y:, cp_l-x: cp_r-x] = img[cp_t:cp_b, cp_l:cp_r]

    return result


def rect_from_center(cx, cy, w, h):
    return (cx - w // 2, cy - h // 2, w, h)
