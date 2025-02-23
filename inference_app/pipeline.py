from threading import Thread
from config import StreamsConfig
from video.streams_manager import StreamsManager
from inference.worker import NetWorker
from concurrent.futures import Future, CancelledError


class Pipeline:
    def __init__(self, streams: list[StreamsConfig]):
        self._thread = Thread(target=self._loop, name='pipeline')
        self.streams_manager = StreamsManager(streams)
        self.worker = NetWorker()

    def next_frame(self):
        return self.streams_manager.next_frame()

    def handle_worker_result(self, future: Future):
        try:
            detections = future.result()
            print(detections)
        except CancelledError:
            pass
        except Exception as ex:
            print('handle_frame_result error:', ex)

    def _process_sources(self):
        frame = self.next_frame()
        if frame is None:
            return None
        if not self.worker.is_busy():
            future = self.worker.submit_frame(frame)
            future.add_done_callback(self.handle_worker_result)

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
