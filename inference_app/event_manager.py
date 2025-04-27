from queue import Queue, Empty
from threading import Thread
from tracker import (
    Tracker,
    TrackEvents,
    TrackCreated,
    TrackUpdated,
    TrackStarted
)
from utils import now_ms


CLEANUP = 1000  # ms


class EventManager:
    def __init__(self, detections: Queue, output: Queue):
        self.detections = detections
        self.output = output
        self._thread = Thread(target=self._loop, name='event-manager')
        self.tracker = Tracker()
        self.last_cleanup = now_ms()

    def run(self):
        self._thread.start()

    def process_tracker_updates(self, updates: list[TrackEvents]):
        events = []

        for upd in updates:
            if isinstance(upd, TrackUpdated):
                print('updated track:', upd.track.id)
                if upd.track.is_started:
                    events.append(upd.track)
            elif isinstance(upd, TrackCreated):
                print('created track', upd.track.id)
            elif isinstance(upd, TrackStarted):
                events.append(upd.track)
                print('started track', upd.track.id)
        return events

    def process_detections(self, detections):
        upds = []
        for detection in detections:
            upds.append(self.tracker.update_track(detection))
        return self.process_tracker_updates(upds)

    def stop(self):
        self._thread.join(5)

    def cleanup(self):
        if now_ms() - self.last_cleanup > CLEANUP:
            try:
                closed_tracks = self.tracker.close_old_tracks()
                self.output.put_nowait(closed_tracks)
            except Exception as exc:
                print(exc)
            finally:
                self.last_cleanup = now_ms()

    def _loop(self):
        while True:
            try:
                detections = self.detections.get_nowait()
                events = self.process_detections(detections)
                if events:
                    self.output.put_nowait(events)
                self.cleanup()
            except Empty:
                self.cleanup()
                continue
