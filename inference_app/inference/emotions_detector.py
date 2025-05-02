from .resnet import ResNet
from .yolo import YOLO
from .insightface import InsightFace

import numpy as np
from dataclasses import dataclass


PREVIEW_SIZE = 256


@dataclass
class EmotionDetection:
    media: str
    pts: int
    left: int
    top: int
    right: int
    bottom: int
    emotion: str
    confidence: float
    preview: bytes
    face_embedding: list[float]


def image_to_bytes(frame):
    return np.array(frame).tobytes('C')


class EmotionsDetector:
    def __init__(self):
        self.detector = YOLO()
        self.classifier = ResNet()
        self.embedder = InsightFace()

    def _crop(self, frame, box):
        return frame.crop(box)

    def execute(self, frame):
        detections = self.detector.run(frame)
        if not detections:
            return []
        result = []
        for detection in detections:
            cropped_frame = self._crop(frame.body, detection.box_ltrb)
            emotion = self.classifier.run(cropped_frame)
            face_embedding = self.embedder.run(cropped_frame)
            result.append(
                EmotionDetection(
                    frame.media,
                    frame.pts,
                    *detection.box_ltrb,
                    confidence=detection.confidence,
                    emotion=emotion,
                    face_embedding=face_embedding,
                    preview=image_to_bytes(cropped_frame.resize((PREVIEW_SIZE, PREVIEW_SIZE)))
                )
            )
        return result
