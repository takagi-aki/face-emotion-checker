from typing import List, Tuple


import cv2
from regex import R


from ..core import IRecognizer


class RecognizerSF:
    """CV2とSFaceで顔認識.

    モデルを以下のリンクからダウンロードする必要あり.
    https://drive.google.com/file/d/1ClK9WiB492c5OZFKveF3XiHCejoOxINW/view  
    """

    def __init__(
        self,
        threshold=0.40,
        **args
    ) -> None:
        self.recongnizer = cv2.FaceRecognizerSF.create(
            "model/opencv/face_recognizer_fast.onnx", "")
        self.saved_face_vec = dict()
        self.threshold = threshold

    def register(self, name, img, face):
        emb = self.embedding(img, face)
        self.saved_face_vec[name] = emb

    def embedding(self, img, face):
        aligned_face = self.recongnizer.alignCrop(img, face)
        emb = self.recongnizer.feature(aligned_face)
        return emb

    def recognize(
        self,
        img: cv2.Mat,
        face,
    ) -> List[Tuple[int, int, int, int]]:
        emb = self.embedding(img, face)

        most_similar = 'unknown'
        high_score = self.threshold
        for name, vec in self.saved_face_vec.items():
            score = self.recongnizer.match(
                emb, vec, cv2.FaceRecognizerSF_FR_COSINE)
            if high_score < score:
                most_similar = name
                high_score = score

        return most_similar, high_score
