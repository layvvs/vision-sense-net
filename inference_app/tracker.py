from typing import Dict, TypeAlias, Union, Any
from snowflake import SnowflakeGenerator
import numpy as np
from dataclasses import dataclass, field

from inference.emotions_detector import EmotionDetection
from utils import now_ms


SNOWFLAKE_EPOCH = 1288834974657
MIN_TRACK_START_QUANTITY = 3
MIN_FACE_DISTANCE = 0.4
MIN_FACE_QUALITY = 9.5
TRACK_TIMEOUT = 30000  # ms
MIN_APPEARANCE_DURATION = 40  # ms

track_ids = SnowflakeGenerator(0, epoch=SNOWFLAKE_EPOCH)


TrackId: TypeAlias = int


def cosine_sim(x, y) -> float:
  dot_products = np.dot(x, y.T)
  norm_products = np.linalg.norm(x) * np.linalg.norm(y)
  return 1 - dot_products / (norm_products + 1e-07)


@dataclass
class DetectionTrack:
    id: int
    media: str
    opened_at: int
    started_at: int | None = None
    started_pts: int | None = None
    updated_at: int | None = None
    updated_pts: int | None = None
    closed_at: int | None = None
    preview: bytes | None = None
    emotions: list[int] = field(default_factory=list)
    embedding: list[float] | None = None
    box: list[float] | None = None
    quality: float = 0
    detections: int = 0
    current_emotion: str | None = None
    emotion_first_appearance: int | None = None
    emotion_last_appearance: int | None = None
    aggregated_emotions: list[dict[str, Any]] = field(default_factory=list)
    last_detection_at: int = 0

    def _get_box_area(self, left, top, right, bottom):
        return (right - left) * (bottom - top)

    @property
    def is_started(self):
        return self.started_at is not None

    def start(self):
        if self.is_started:
            raise ValueError('already started')
        now = now_ms()
        self.started_at = now
        self.updated_at = now

    def _aggregate_emotions(self, emotion, timestamp):
        if self.current_emotion is None:
            self.current_emotion = emotion
            self.emotion_first_appearance = timestamp
            self.emotion_last_appearance = timestamp + MIN_APPEARANCE_DURATION
            return
        if self.current_emotion == emotion:
            self.emotion_last_appearance = timestamp
        else:
            self.finalize_emotions_aggregation()
            self._aggregate_emotions(emotion, timestamp)

    def finalize_emotions_aggregation(self):
        emotion = {
            self.current_emotion: {
                'first_appearance': self.emotion_first_appearance,
                'last_appearance': self.emotion_last_appearance
            }
        }
        self.aggregated_emotions.append(emotion)
        self.current_emotion = None
        self.emotion_first_appearance = None
        self.emotion_last_appearance = None

    def add_detection(self, detection: EmotionDetection):
        self.detections += 1
        self.last_detection_at = now_ms()
        self.updated_at = self.last_detection_at
        emotion = detection.emotion
        self.emotions.append(emotion)
        self._aggregate_emotions(emotion, self.last_detection_at)
        quality = detection.confidence * self._get_box_area(
            detection.left,
            detection.top,
            detection.right,
            detection.bottom
        )
        if quality > self.quality:
            self.quality = quality
            self.preview = detection.preview
            self.embedding = detection.face_embedding

    def close(self):
        self.closed_at = self.updated_at

    @classmethod
    def create(self, media) -> 'DetectionTrack':
        id = next(track_ids)
        return DetectionTrack(
            id=id,
            media=media,
            opened_at=now_ms()
        )


@dataclass
class TrackEventOk:
    track: DetectionTrack


class TrackCreated(TrackEventOk):
    ...


class TrackUpdated(TrackEventOk):
    ...


class TrackStarted(TrackEventOk):
    ...


@dataclass
class TrackEventError:
    ...

class TrackErrorQuality(TrackEventError):
    ...


TrackEvents = Union[TrackCreated, TrackUpdated, TrackStarted, TrackErrorQuality]


class Tracker:
    _tracks: Dict[TrackId, DetectionTrack]

    def __init__(self):
        self._tracks = {}

    def _find_track(self, vector: list[float]) -> DetectionTrack:
        tracks = self._tracks.values()
        for track in tracks:
            sim = cosine_sim(track.embedding, vector)
            if sim < MIN_FACE_DISTANCE:
                return track
            else:
                return

    def update_track(self, detection: EmotionDetection):
        embedding = detection.face_embedding
        if np.linalg.norm(embedding) < MIN_FACE_QUALITY:
            return TrackErrorQuality()
        if (track := self._find_track(detection.face_embedding)):
            track.add_detection(detection)
            if not track.is_started and track.detections >= MIN_TRACK_START_QUANTITY:
                track.start()
                return TrackStarted(track)
            return TrackUpdated(track)
        else:
            track = self._create_track(detection)
            track.add_detection(detection)
            self._tracks[track.id] = track
            return TrackCreated(track)

    def _create_track(self, detection: EmotionDetection):
        return DetectionTrack.create(
            detection.media,
        )

    def close_track(self, track_key: TrackId):
        track = self._tracks.pop(track_key)
        track.finalize_emotions_aggregation()
        track.close()
        print('closed track', track.id)
        return track

    def close_old_tracks(self):
        closed_tracks = []
        for key, track in list(self._tracks.items()):
            if now_ms() - track.last_detection_at > TRACK_TIMEOUT:
                track = self.close_track(key)
                if track.is_started: closed_tracks.append(track)
        return closed_tracks
