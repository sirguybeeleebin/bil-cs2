from pathlib import Path
import redis
from telegram import Bot

BASE_DIR = Path(__file__).resolve().parent.parent

# -----------------------------
# Basic
# -----------------------------
SECRET_KEY = "django-insecure-CHANGE-THIS-IN-PRODUCTION"
DEBUG = True
ALLOWED_HOSTS = ["*"]

# -----------------------------
# Installed Apps
# -----------------------------
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",

    # Third-party
    "rest_framework",

    # Local apps
    "app",
    
    "rest_framework_simplejwt.token_blacklist",
    "drf_spectacular",
]

# -----------------------------
# REST Framework + JWT + OpenAPI
# -----------------------------
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": (
        "rest_framework_simplejwt.authentication.JWTAuthentication",
    ),
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
}

SPECTACULAR_SETTINGS = {
    "TITLE": "Prediction API",
    "DESCRIPTION": "API для прогнозов с JWT",
    "VERSION": "1.0.0",
    "SERVE_INCLUDE_SCHEMA": False,
}

# -----------------------------
# Middleware
# -----------------------------
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
]

# -----------------------------
# URLs / WSGI
# -----------------------------
ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"

# -----------------------------
# Database (PostgreSQL)
# -----------------------------
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": "appdb",
        "USER": "pg",
        "PASSWORD": "pg",
        "HOST": "db",
        "PORT": "5432",
    }
}

# -----------------------------
# Redis + Celery
# -----------------------------
REDIS_HOST = "redis"
REDIS_PORT = 6379

CELERY_BROKER_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"
CELERY_RESULT_BACKEND = f"redis://{REDIS_HOST}:{REDIS_PORT}/1"

CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"

# -----------------------------
# Static files
# -----------------------------
STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

# -----------------------------
# Telegram + Redis Pub/Sub
# -----------------------------
TELEGRAM_TOKEN = "YOUR_TELEGRAM_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"
TELEGRAM_BOT = Bot(token=TELEGRAM_TOKEN)

REDIS_CLIENT = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
