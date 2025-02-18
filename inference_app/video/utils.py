from typing import NamedTuple


class Frame(NamedTuple):
    width: int
    height: int
    pts: int = None
    body: None = None

    @staticmethod
    def from_av_frame(frame, timebase=None):
        timebase = timebase or frame.time_base._denominator
        # TODO: add rgb converting + body as np.array
        return Frame(
            width=frame.width,
            height=frame.height,
            pts=1000 * int(frame.pts) // timebase,
            body=frame
        )
