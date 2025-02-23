from typing import NamedTuple
from av.video import VideoFrame


class Frame(NamedTuple):
    width: int
    height: int
    pts: int = None
    body: None = None

    @staticmethod
    def from_av_frame(frame: VideoFrame, timebase=None):
        timebase = timebase or frame.time_base._denominator
        return Frame(
            width=frame.width,
            height=frame.height,
            pts=1000 * int(frame.pts) // timebase,
            body=frame.to_rgb().to_image()
        )
