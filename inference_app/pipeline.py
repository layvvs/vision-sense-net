from config import StreamsConfig
from video.streams_manager import StreamsManager
from inference.worker import NetWorker
from event_manager import EventManager

from threading import Thread
from concurrent.futures import Future, CancelledError
from queue import Queue


class Pipeline:
    def __init__(
            self,
            streams: list[StreamsConfig],
            output: Queue
    ):
        self._thread = Thread(target=self._loop, name='pipeline')
        self.streams_manager = StreamsManager(streams)
        self.worker = NetWorker()
        self.output = output
        self.detections = Queue(maxsize=20)
        self.event_manager = EventManager(self.detections, self.output)

    def next_frame(self):
        return self.streams_manager.next_frame()

    def _handle_worker_result(self, future: Future):
        try:
            detections = future.result()
            if detections:
                self.detections.put_nowait(detections)
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
            future.add_done_callback(self._handle_worker_result)

    def _loop(self):
        while True:
            self._process_sources()

    def run(self):
        self.streams_manager.run()
        print('StreamsManager started')
        self.event_manager.run()
        print('EventManager started')
        self._thread.start()

    def stop(self):
        self.streams_manager.stop()
        self.event_manager.stop()
        print('EventManager stopped')
        self._thread.join(5)

    def __enter__(self):
        self.run()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop()
