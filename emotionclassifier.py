

class IEmotionClassifier:

    def classify(self):
        #
        # 顔を検出する処理
        #
        pass


def get_classifier(name:str) -> IEmotionClassifier:
    return IEmotionClassifier()
