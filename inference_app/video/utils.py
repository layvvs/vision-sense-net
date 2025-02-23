from typing import NamedTuple
from av.video import VideoFrame


class Frame(NamedTuple):
    media: str
    width: int
    height: int
    pts: int = None
    body: None = None

    @staticmethod
    def from_av_frame(frame: VideoFrame, media: str, timebase=None):
        timebase = timebase or frame.time_base._denominator
        return Frame(
            media=media,
            width=frame.width,
            height=frame.height,
            pts=1000 * int(frame.pts) // timebase,
            body=frame.to_rgb().to_image()
        )
