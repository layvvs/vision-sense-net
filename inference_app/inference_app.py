from config import ServiceConfig
from pipeline import Pipeline
import threading
from queue import Queue, Empty


class InferenceApp:
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.events = Queue(maxsize=20)
        self.pipeline = Pipeline(self.config.streams, self.events)
        threading.current_thread().name = 'inference-app'

    def _loop(self):
        while True:
            try:
                self.events.get_nowait()
            except Empty:
                continue

    def run(self):
        with self.pipeline:
            self._loop()
