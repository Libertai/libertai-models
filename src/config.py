import json
import os

from dotenv import load_dotenv
from pydantic import BaseModel


class ModelConfig(BaseModel):
    id: str
    url: str
    allowed_paths: list[str]


class _Config:
    BACKEND_URL: str
    MODEL_CONFIGS: dict[str, ModelConfig]
    API_PUBLIC_KEY: str

    def __init__(self):
        load_dotenv()

        self.BACKEND_URL = os.getenv("BACKEND_URL")
        self.API_PUBLIC_KEY = os.getenv("API_PUBLIC_KEY")

        # Load model configurations from model names in environment variable
        model_names = os.getenv("MODELS", "").split(",")
        self.MODEL_CONFIGS = {}

        for model_name in model_names:
            model_name = model_name.strip()
            if not model_name:
                continue

            model_config_path = f"./data/{model_name}.json"

            try:
                with open(model_config_path) as f:
                    model_data = json.load(f)
                model_config = ModelConfig(**model_data)
                self.MODEL_CONFIGS[model_config.id] = model_config

            except json.JSONDecodeError as error:
                print(f"Error on {model_config_path} config file")
                print(error)
            except FileNotFoundError:
                print(f"Config file not found: {model_config_path}")


config = _Config()
