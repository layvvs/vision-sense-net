from dataclasses import dataclass


@dataclass
class EmotionDetection:
    left: int
    top: int
    right: int
    bottom: int
    emotion_class: str
    confidense: float
