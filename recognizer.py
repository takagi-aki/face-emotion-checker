

class IRecognizer:

    def recognize(self):
        #
        # 顔を検出する処理
        #
        pass


def get_recognizer(name:str) -> IRecognizer:
    return IRecognizer()
