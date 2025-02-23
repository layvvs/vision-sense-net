from .stream_reader import StreamReader
from config import StreamsConfig
import time
from dataclasses import dataclass


@dataclass
class Stream:
    media_id: str
    reader: StreamReader
    last_frame_at: int = 0


class StreamsManager:
    def __init__(self, streams: list[StreamsConfig]):
        self._streams = [
            Stream(media_id=stream.name, reader=StreamReader(stream.name, stream.url))
            for stream in streams
        ]

    def run(self):
        for stream in self._streams:
            stream.reader.run()

    def stop(self):
        for stream in self._streams:
            stream.reader.stop()

    def next_frame(self):
        if not self._streams:
            return
        for stream in sorted(self._streams, key=lambda it: it.last_frame_at):
            try:
                frame = next(stream.reader)
                if frame is None:
                    continue
            except StopIteration:
                stream.reader.stop()
                self._streams.remove(stream)
                continue
            stream.last_frame_at = time.time()
            return frame
