from threading import Thread
from config import StreamsConfig
from stream_reader import StreamReader
import time


class Pipeline:
    def __init__(self, streams: list[StreamsConfig]):
        self._thread = Thread(target=self._loop, name='pipeline')
        # TODO: Manage streams in another class
        self._streams = [StreamReader(stream.name, stream.url) for stream in streams]

    def next_frame(self):
        if not self._streams:
            return
        for stream in sorted(self._streams, key=lambda it: it.last_frame_at):
            try:
                frame = next(stream)
                if frame is None:
                    continue
            except StopIteration:
                print(f'{stream.media} error')
                self._streams.remove(stream)
                continue
            stream.last_frame_at = time.time()
            return frame

    def _process_sources(self):
        if self._streams:
            frame = self.next_frame()
            if frame is None:
                return None
            print(frame)

    def _loop(self):
        while True:
            self._process_sources()

    def run(self):
        for stream in self._streams:
            stream.run()
        print('Streams started')
        print('pipe Start')
        self._thread.run()
        print('pipe Started')

    def stop(self):
        print('Stop')
        for stream in self._streams:
            stream.stop()
        self._thread.join(5)

    def __enter__(self):
        self.run()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop()
