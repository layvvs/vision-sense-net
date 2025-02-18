from threading import Thread
from config import StreamsConfig
from video.streams_manager import StreamsManager


class Pipeline:
    def __init__(self, streams: list[StreamsConfig]):
        self._thread = Thread(target=self._loop, name='pipeline')
        self.streams_manager = StreamsManager(streams)

    def next_frame(self):
        return self.streams_manager.next_frame()

    def _process_sources(self):
        frame = self.next_frame()
        if frame is None:
            return None

    def _loop(self):
        while True:
            self._process_sources()

    def run(self):
        self.streams_manager.run()
        print('Streams started')
        print('pipe Start')
        self._thread.start()
        print('pipe Started')

    def stop(self):
        print('Stop')
        self.streams_manager.stop()
        self._thread.join(5)

    def __enter__(self):
        self.run()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop()
