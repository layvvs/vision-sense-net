from dataclasses import dataclass, asdict
from base64 import b64encode
from io import BytesIO
from PIL import Image
import numpy as np
from collections import Counter, defaultdict

from tracker import DetectionTrack

PREVIEW_SIZE = 256


@dataclass
class Event:
    id: int
    opened_at: int
    started_at: int
    updated_at: int
    closed_at: int
    preview: str
    emotions: list[str]
    emotions_hist: dict
    most_common_emotion: str
    most_lasting_emotion: str
    most_common_emotion_duration: int
    emotions_aggregation: list
    emotions_duration: dict


def encode_preview(preview):
    with BytesIO() as buf:
        image = Image.fromarray(np.frombuffer(preview, dtype=np.uint8).reshape((PREVIEW_SIZE, PREVIEW_SIZE, 3)))
        image.save(buf, format="JPEG")
        return b64encode(buf.getvalue()).decode('utf-8')


# def get_emotions_statistics(emotions, aggregated_emotions):
#     emotions_hist = Counter(emotions)
#     most_common_emotion, *_ = emotions_hist.most_common(1)[0]
#     most_common_emotion_duration = 0
#     emotions_duration = defaultdict(int)
#     for emotion in aggregated_emotions:
#         emotion_label, *_ = list(emotion.keys())
#         dur = emotion[emotion_label]['last_appearance'] - emotion[emotion_label]['first_appearance']
#         emotions_duration[emotion_label] += dur
#         if emotion_label == most_common_emotion:
#             most_common_emotion_duration += dur
#     most_lasting_emotion, *_ = sorted(emotions_duration, key=lambda x: emotions_duration[x], reverse=True)

#     return (
#         emotions_hist,
#         most_common_emotion,
#         most_common_emotion_duration,
#         emotions_duration,
#         most_lasting_emotion
#     )


def get_emotions_statistics(emotions, aggregated_emotions):
    emotions_hist = Counter(emotions)
    
    # Handle empty emotions_hist
    if not emotions_hist:
        most_common_emotion = None
        most_common_emotion_duration = 0
    else:
        most_common_emotion, _ = emotions_hist.most_common(1)[0]
        most_common_emotion_duration = 0

    emotions_duration = defaultdict(int)
    for emotion in aggregated_emotions:
        emotion_label, *_ = list(emotion.keys())
        dur = emotion[emotion_label]['last_appearance'] - emotion[emotion_label]['first_appearance']
        emotions_duration[emotion_label] += dur
        if emotion_label == most_common_emotion:
            most_common_emotion_duration += dur

    lasting_emotions = sorted(emotions_duration, key=lambda x: emotions_duration[x], reverse=True)
    most_lasting_emotion = lasting_emotions[0] if lasting_emotions else None

    return (
        dict(emotions_hist),
        most_common_emotion,
        most_common_emotion_duration,
        dict(emotions_duration),
        most_lasting_emotion
    )


def map_tracker(track: DetectionTrack):
    try:
        emotions = track.emotions
        aggregated_emotions = track.aggregated_emotions
        emotions_hist, most_common_emotion, most_common_emotion_duration, emotions_duration, most_lasting_emotion = get_emotions_statistics(emotions, aggregated_emotions)
        eve = Event(
            id=track.id,
            opened_at=track.opened_at,
            started_at=track.started_at,
            updated_at=track.updated_at,
            closed_at=track.closed_at,
            preview=encode_preview(track.preview),
            emotions=track.emotions,
            emotions_hist=emotions_hist,
            most_common_emotion=most_common_emotion,
            most_common_emotion_duration=most_common_emotion_duration,
            emotions_aggregation=aggregated_emotions,
            emotions_duration=emotions_duration,
            most_lasting_emotion=most_lasting_emotion
        )
        return eve
    except Exception as e:
        print(f"[map_tracker] Failed to map track {track.id}: {e}")
        return None
    # emotions = track.emotions
    # aggregated_emotions = track.aggregated_emotions
    # emotions_hist, most_common_emotion, most_common_emotion_duration, emotions_duration, most_lasting_emotion = get_emotions_statistics(emotions, aggregated_emotions)

    # return Event(
    #     id=track.id,
    #     opened_at=track.opened_at,
    #     started_at=track.started_at,
    #     updated_at=track.updated_at,
    #     closed_at=track.closed_at,
    #     preview=encode_preview(track.preview),
    #     emotions=track.emotions,
    #     emotions_hist=emotions_hist,
    #     most_common_emotion=most_common_emotion,
    #     most_common_emotion_duration=most_common_emotion_duration,
    #     emotions_aggregation=aggregated_emotions,
    #     emotions_duration=emotions_duration,
    #     most_lasting_emotion=most_lasting_emotion
    # )
