import json
import os

from dotenv import load_dotenv
from pydantic import BaseModel


class ModelConfig(BaseModel):
    id: str
    url: str
    completion_paths: list[str]


class _Config:
    BACKEND_API_URL: str
    MODEL_CONFIG: ModelConfig
    BACKEND_PUBLIC_KEY: str

    def __init__(self):
        load_dotenv()

        self.BACKEND_API_URL = os.getenv("BACKEND_API_URL")
        self.BACKEND_PUBLIC_KEY = os.getenv("BACKEND_PUBLIC_KEY")

        # Load model configuration from environment variable or file
        model_config = os.getenv("MODEL_CONFIG")
        self.MODELS = {}

        try:
            with open(model_config) as f:
                model_data = json.load(f)
            self.MODEL_CONFIG = ModelConfig(**model_data)

        except json.JSONDecodeError as error:
            print(f"Error on {model_config} config file")
            print(error)


config = _Config()
