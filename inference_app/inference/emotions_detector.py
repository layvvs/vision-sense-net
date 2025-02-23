from .resnet import ResNet
from .yolo import Yolo
from .utils import EmotionDetection


class EmotionsDetector:
    def __init__(self):
        self.detector = Yolo()
        self.classifier = ResNet()

    def execute(self, frame):
        detections = self.detector.run(frame)
        if not detections:
            return []

        result = []
        for detection in detections:
            emotion_class = self.classifier.run(frame, detection.box_ltrb)
            result.append(
                EmotionDetection(*detection.box_ltrb, confidense=detection.confidense, emotion_class=emotion_class)
            )
        return result
