import os

from dotenv import load_dotenv


def get_config() -> dict:
    load_dotenv()
    return {
        "BOT_TOKEN": os.getenv("TELEGRAM_TOKEN"),
        "API_BASE_URL": os.getenv("API_BASE_URL", "http://localhost:8000/api/v1"),
    }
