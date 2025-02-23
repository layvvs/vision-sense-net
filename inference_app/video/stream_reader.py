import av
from threading import Thread, Event
from collections import deque
from video.utils import Frame


AV_OPEN_TIMEOUT = 20
AV_READ_TIMEOUT = 5

av_timeout = (AV_OPEN_TIMEOUT, AV_READ_TIMEOUT)


class StreamReader:
    def __init__(self, media: str, stream_url: str):
        self.media = media
        self.stream_url = stream_url
        self.decoded_frames = deque(maxlen=10)
        self.thread = Thread(target=self._loop, name=f'{media}-stream-reader')
        self._last_frame_at = 0
        self.stream_done = Event()

    @property
    def last_frame_at(self):
        return self._last_frame_at

    @last_frame_at.setter
    def last_frame_at(self, value):
        self._last_frame_at = value

    def _read_stream(self):
        container = av.open(self.stream_url, timeout=av_timeout)
        for frame in container.decode(video=0):
            yield Frame.from_av_frame(frame)
        self.stream_done.set()

    def _loop(self):
        for frame in self._read_stream():
            if frame is None:
                continue
            self.decoded_frames.append(frame)

    def last_decoded(self):
        if len(self.decoded_frames):
            return self.decoded_frames.popleft()
        return

    def run(self):
        self.thread.start()

    def stop(self):
        print('Stream', self.media, 'is done')
        self.thread.join()

    def __next__(self):
        if self.stream_done.is_set():
            raise StopIteration
        frame = self.last_decoded()
        return frame

    def __iter__(self):
        return self
