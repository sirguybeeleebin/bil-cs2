import os
from pathlib import Path
import httpx

# --------------------------------------
# Paths & Base
# --------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# --------------------------------------
# Security & Debug
# --------------------------------------
SECRET_KEY = "django-insecure-CHANGE-THIS-IN-PRODUCTION"
DEBUG = True
ALLOWED_HOSTS = ["*"]

# --------------------------------------
# URL & WSGI/ASGI
# --------------------------------------
ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

# --------------------------------------
# Static files
# --------------------------------------
STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

# --------------------------------------
# Database
# --------------------------------------
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": "cs2_db",
        "USER": "cs2_user",
        "PASSWORD": "cs2_password",
        "HOST": "localhost",
        "PORT": "5433",
    }
}

# --------------------------------------
# Redis
# --------------------------------------
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_USER = "default"
REDIS_PASSWORD = "redis_password"
REDIS_DB = 0

# --------------------------------------
# Celery
# --------------------------------------
CELERY_BROKER_URL = f"redis://{REDIS_USER}:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"
CELERY_RESULT_BACKEND = f"redis://{REDIS_USER}:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/1"
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_TIMEZONE = "UTC"

# --------------------------------------
# REST Framework
# --------------------------------------
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": (
        "rest_framework_simplejwt.authentication.JWTAuthentication",
    ),
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
}

# --------------------------------------
# Fluentd
# --------------------------------------
FLUENTD_URL = "http://localhost:9880/logs"

transport = httpx.HTTPTransport(retries=3)
HTTP_CLIENT = httpx.Client(timeout=2.0, transport=transport)

# --------------------------------------
# Installed Apps
# --------------------------------------
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "rest_framework_simplejwt",
    "rest_framework_simplejwt.token_blacklist",
    "drf_spectacular",
    "app",
    "channels",
    "django_celery_beat",
]

# --------------------------------------
# Logging
# --------------------------------------
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "fmt": "%(asctime)s %(levelname)s %(name)s %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "json",
        },
    },
    "loggers": {
        "django.request_json": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
    },
}

# --------------------------------------
# Middleware
# --------------------------------------
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "app.middlewares.JsonLoggingMiddleware",  
]

# --------------------------------------
# ML Paths & Settings
# --------------------------------------
PATH_TO_GAMES_RAW_DIR = BASE_DIR / "data" / "games_raw"
PATH_TO_ML_RESULTS_DIR = BASE_DIR / "data" / "ml_results"

ML_PIPELINE_SETTINGS = {
    "TEST_SIZE": 100,
    "N_SPLITS": 10,
    "N_ITERS": 10,
    "RANDOM_STATE": 42,
}

# --------------------------------------
# Channels
# --------------------------------------
CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels_redis.core.RedisChannelLayer",
        "CONFIG": {
            "hosts": [(REDIS_HOST, REDIS_PORT)],
        },
    },
}

# --------------------------------------
# drf-spectacular
# --------------------------------------
SPECTACULAR_SETTINGS = {
    "TITLE": "Prediction API",
    "DESCRIPTION": "API для ML предсказаний",
    "VERSION": "1.0.0",
}

# --------------------------------------
# Templates
# --------------------------------------
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],  
        "APP_DIRS": True,  
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    }
]
