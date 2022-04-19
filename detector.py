

class IDetector:

    def detect(self, hogehoge):
        #
        # 顔を検出する処理
        #
        pass


def get_detector(name: str) -> IDetector:
    return IDetector()
