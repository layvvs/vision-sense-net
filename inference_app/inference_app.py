from config import ServiceConfig
from pipeline import Pipeline
import threading


class InferenceApp:
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.pipeline = Pipeline(self.config.streams)
        threading.current_thread().name = 'inference-app'

    def _loop(self):
        while True:
            ...

    def run(self):
        with self.pipeline:
            self._loop()
