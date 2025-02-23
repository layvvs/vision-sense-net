from pydantic import BaseModel
import yaml


class StreamsConfig(BaseModel):
    name: str
    url: str


class WebConfig(BaseModel):
    host: str
    port: int


class ServiceConfig(BaseModel):
    web: WebConfig
    streams: list[StreamsConfig]


def get_config():
    with open("config.yml", "r") as file:
        raw_config = yaml.safe_load(file)

    return ServiceConfig.model_validate(raw_config)
