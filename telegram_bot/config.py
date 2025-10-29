import os

BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
