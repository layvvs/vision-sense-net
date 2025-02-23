from dataclasses import dataclass


@dataclass
class EmotionDetection:
    media: str
    pts: int
    left: int
    top: int
    right: int
    bottom: int
    emotion_class: str
    confidence: float
    preview: bytes
    face_embedding: list[float]
