import os
from pathlib import Path

from dotenv import load_dotenv

# Подгружаем .env перед настройкой Django
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# На всякий случай можно установить дефолтный SECRET_KEY
os.environ.setdefault("SECRET_KEY", "django-insecure-dev-key")
