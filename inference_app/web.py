from config import get_config
from config import ServiceConfig
from handlers import handle_list_events
from database.database import DatabaseConnection

from sanic import Sanic, Blueprint
from sanic_cors import CORS
import asyncio


config: ServiceConfig  = get_config()

app = Sanic(name='inference')
CORS(app)

app.ctx.database = DatabaseConnection()

asyncio.run(app.ctx.database.connect())

api = Blueprint('api')

api.add_route(handle_list_events, "/events", ["GET"], name='list_events')
app.blueprint(api, url_prefix='/inference/api')

app.run(host=config.web.host, port=config.web.port, workers=1, single_process=True)
