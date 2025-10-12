from pathlib import Path
from celery.schedules import crontab
import os

# -------------------------------------------------------------------
# BASE SETTINGS
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# -------------------------------------------------------------------
# SECURITY
# -------------------------------------------------------------------
SECRET_KEY = "django-insecure-dev-key"
DEBUG = True
ALLOWED_HOSTS = ["localhost", "127.0.0.1", "0.0.0.0"]

# -------------------------------------------------------------------
# DJANGO APPS
# -------------------------------------------------------------------
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django_celery_beat",
    "backend",
]

# -------------------------------------------------------------------
# MIDDLEWARE
# -------------------------------------------------------------------
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

# -------------------------------------------------------------------
# URL & WSGI
# -------------------------------------------------------------------
ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"

# -------------------------------------------------------------------
# TEMPLATES
# -------------------------------------------------------------------
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
    },
]

# -------------------------------------------------------------------
# DATABASES (PostgreSQL)
# -------------------------------------------------------------------
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": "cs2_db",
        "USER": "cs2_user",
        "PASSWORD": "cs2_password",
        "HOST": "localhost",
        "PORT": "5432",
    }
}

# -------------------------------------------------------------------
# ML PIPELINE DIRECTORIES
# -------------------------------------------------------------------
GAMES_RAW_DIR = BASE_DIR / "data" / "games_raw"       # Raw JSON files
GAMES_VALID_DIR = BASE_DIR / "data" / "games_valid"   # Validated games
ML_INPUT_DIR = BASE_DIR / "data" / "ml_input"         # Train/test splits
ML_RESULTS_DIR = BASE_DIR / "data" / "ml_results"     # ML pipeline outputs

# Ensure directories exist
for path in [GAMES_RAW_DIR, GAMES_VALID_DIR, ML_INPUT_DIR, ML_RESULTS_DIR]:
    os.makedirs(path, exist_ok=True)

# -------------------------------------------------------------------
# ML TRAIN/TEST SPLIT
# -------------------------------------------------------------------
TEST_SIZE = 100

# -------------------------------------------------------------------
# AUTH
# -------------------------------------------------------------------
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# -------------------------------------------------------------------
# INTERNATIONALIZATION
# -------------------------------------------------------------------
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

# -------------------------------------------------------------------
# STATIC & MEDIA
# -------------------------------------------------------------------
STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

# -------------------------------------------------------------------
# CELERY CONFIGURATION (Redis)
# -------------------------------------------------------------------
CELERY_BROKER_URL = "redis://localhost:6379/0"
CELERY_RESULT_BACKEND = "redis://localhost:6379/0"
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_TIMEZONE = "UTC"
CELERY_TASK_TRACK_STARTED = True
CELERY_TASK_TIME_LIMIT = 30 * 60  # 30 minutes

CELERY_BEAT_SCHEDULE_FILENAME = str(BASE_DIR / "celerybeat-schedule")
CELERY_BEAT_SCHEDULE = {
    "daily-full-pipeline": {
        "task": "backend.tasks.run_ml_pipeline",
        "schedule": crontab(hour=0, minute=0),  # runs daily at midnight UTC
    },
}

# -------------------------------------------------------------------
# DEFAULT AUTO FIELD
# -------------------------------------------------------------------
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
