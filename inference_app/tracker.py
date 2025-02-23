from typing import Dict, TypeAlias, Union
from snowflake import SnowflakeGenerator
import numpy as np
from dataclasses import dataclass, field

from inference.utils import EmotionDetection
from utils import now_ms


SNOWFLAKE_EPOCH = 1288834974657
MIN_TRACK_START_QUANTITY = 3
MIN_FACE_DISTANCE_ADD = 0.5
MAX_FACE_DISTANCE_CREATE = 0.3
TRACK_TIMEOUT = 30000  # ms

track_ids = SnowflakeGenerator(0, epoch=SNOWFLAKE_EPOCH)


TrackId: TypeAlias = int


def cosine_sim(x, y):
    return np.dot(x, y.T) / (np.linalg.norm(x) * np.linalg.norm(y))


@dataclass
class DetectionTrack:
    id: int
    media: str
    opened_at: int
    updated_at: int | None = None
    closed_at: int | None = None
    preview: bytes | None = None
    emotions: list[int] = field(default_factory=list)
    embedding: list[float] | None = None
    box: list[float] | None = None
    quality: float | None = 0
    detections: int = 0
    last_detection_at: int = 0

    def _get_box_area(self, left, top, right, bottom):
        return (right - left) * (bottom - top)

    def add_detection(self, detection: EmotionDetection):
        self.detections += 1
        self.last_detection_at = now_ms()
        self.updated_at = detection.pts
        self.emotions.append(detection.emotion_class)
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
        self.closed_at = now_ms()

    @classmethod
    def create(self, media, opened_at) -> 'DetectionTrack':
        id = next(track_ids)
        return DetectionTrack(
            id=id,
            media=media,
            opened_at=opened_at
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


class TrackErrorDistance(TrackEventError):
    ...


TrackEvents = Union[TrackCreated, TrackUpdated, TrackStarted, TrackErrorDistance]


class Tracker:
    _tracks: Dict[TrackId, DetectionTrack]

    def __init__(
        self,
        start_quantity=MIN_TRACK_START_QUANTITY,
        face_distance_add=MIN_FACE_DISTANCE_ADD,
        face_distance_create=MAX_FACE_DISTANCE_CREATE,
        timeout=TRACK_TIMEOUT
    ):
        self._tracks = {}
        self.start_quantity = start_quantity
        self.min_face_distance_add = face_distance_add
        self.min_face_distance_create = face_distance_create
        self.timeout = timeout

    def _find_track(self, vector: list[float]) -> DetectionTrack:
        tracks = self._tracks.values()
        for track in tracks:
            sim = cosine_sim(track.embedding, vector)
            if sim > self.min_face_distance_add:
                return track
            elif sim < self.min_face_distance_create:
                return TrackErrorDistance()
            else:
                return

    def update_track(self, detection: EmotionDetection):
        if (track := self._find_track(detection.face_embedding)):
            if isinstance(track, TrackErrorDistance):
                return track
            track.add_detection(detection)
            if track.detections >= self.start_quantity:
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
            opened_at=detection.pts
        )

    def close_track(self, track_key: TrackId):
        track = self._tracks.pop(track_key)
        track.close()
        return track

    def close_old_tracks(self):
        closed_tracks = []
        for key, track in list(self._tracks.items()):
            if now_ms() - track.last_detection_at > self.timeout:
                track = self.close_track(key)
                closed_tracks.append(track)
        return closed_tracks
