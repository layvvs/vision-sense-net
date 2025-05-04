import threading
from queue import Queue, Empty
from event_mapper import map_tracker
import asyncio
from dataclasses import asdict

from config import ServiceConfig
from pipeline import Pipeline
from database.database import DatabaseConnection


#  Еще раз про ивент луп почитать и понять, правильно ли тут все делается
#  Добавить логгинг нормальный
#  Больше try/except
class InferenceApp:
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.events = Queue(maxsize=20)
        self.pipeline = Pipeline(self.config.streams, self.events)
        self.database = DatabaseConnection()
        threading.current_thread().name = 'inference-app'

    async def save_event_to_database(self, event):
        await self.database.set(event.id, str(asdict(event)))

    async def _process_events(self):
        while True:
            try:
                events = self.events.get_nowait()
                for event in events:
                    mapped_event = map_tracker(event)
                    await self.save_event_to_database(mapped_event)
            except Empty:
                await asyncio.sleep(0.1)
            except Exception as exc:
                print('Exception occurred:', exc)

    async def _main(self):
        await self.database.connect()
        await self._process_events()

    def run(self):
        with self.pipeline:
            asyncio.run(self._main())
