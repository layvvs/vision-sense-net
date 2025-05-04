import redis.asyncio as redis
import os
import orjson
from dotenv import load_dotenv

load_dotenv()

USERNAME = os.getenv("REDIS_USER")
PASSWORD = os.getenv("REDIS_USER_PASSWORD")
TABLENAME = 'events'


class DatabaseConnection:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DatabaseConnection, cls).__new__(cls)
        return cls._instance

    def __init__(self, host='0.0.0.0', port=6380, db=0, username=USERNAME, password=PASSWORD):
        if not hasattr(self, '_initialized'):
            self.host = host
            self.port = port
            self.db = db
            self.username = username
            self.password = password
            self._redis = None
            self._initialized = True

    async def connect(self):
        self._redis = redis.StrictRedis(
            host=self.host,
            port=self.port,
            db=self.db,
            username=self.username,
            password=self.password,
            decode_responses=True,
        )
        print('connected to database')

    async def get_records(self):
        records = []
        _, partial_keys = await self._redis.scan(count=100)

        for key in partial_keys:
            data = await self._redis.get(key)
            if data:
                records.append(data)

        return records

    async def set(self, key: str, value: str):
        await self._redis.set(name=key, value=value)
