from .emotions_detector import EmotionsDetector
from concurrent.futures import ThreadPoolExecutor


class NetWorker(ThreadPoolExecutor):
    def __init__(self, max_workers=2):
        self.module = EmotionsDetector()
        super().__init__(max_workers=max_workers, thread_name_prefix='emitions-detector')

    def _process_frame(self, frame):
        try:
            return self.module.execute(frame)
        except Exception as ex:
            print("_process_frame error:", ex)
            raise

    def submit_frame(self, frame):
        fut = self.submit(self._process_frame, frame)
        return fut

    def is_busy(self):
        return self._work_queue.qsize() > 2 * self._max_workers
