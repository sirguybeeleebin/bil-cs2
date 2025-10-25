# main.py
import os
from pathlib import Path
from dotenv import load_dotenv
import redis
from celery import Celery
from celery.schedules import crontab
import httpx

from ml.ml_pipeline import run_ml_pipeline
from tasks_factory import make_train_and_publish_task
from tasks_update import make_update_dictionaries_task  # ваша фабрика задачи

load_dotenv()

ML_RESULTS_PATH = os.getenv("ML_RESULTS_PATH", "data/ml_results")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_CHANNEL = os.getenv("REDIS_CHANNEL", "ml_updates")
CELERY_BROKER_URL = os.getenv(
    "CELERY_BROKER_URL", f"redis://{REDIS_HOST}:{REDIS_PORT}/1"
)
CELERY_BACKEND_URL = os.getenv(
    "CELERY_BACKEND_URL", f"redis://{REDIS_HOST}:{REDIS_PORT}/2"
)

Path(ML_RESULTS_PATH).mkdir(parents=True, exist_ok=True)

celery_app = Celery(
    "tasks",
    broker=CELERY_BROKER_URL,
    backend=CELERY_BACKEND_URL,
)

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

# Train and publish task
train_and_publish_task = make_train_and_publish_task(
    celery_app=celery_app,
    ml_results_path=ML_RESULTS_PATH,
    redis_client=redis_client,
    redis_channel=REDIS_CHANNEL,
    run_ml_pipeline_func=run_ml_pipeline,
)

# Async HTTP client for update dictionaries task
async_client = httpx.AsyncClient(timeout=10.0)
update_dictionaries_task = make_update_dictionaries_task(
    celery_app=celery_app,
    http_client=async_client,
)

# Celery beat schedule
celery_app.conf.beat_schedule = {
    "train-every-midnight": {
        "task": train_and_publish_task.name,
        "schedule": crontab(hour=0, minute=0),
    },
    "update-dictionaries-every-hour": {
        "task": update_dictionaries_task.name,
        "schedule": crontab(minute=0),
    },
}
celery_app.conf.timezone = "UTC"

@celery_app.on_after_finalize.connect
def startup(sender, **kwargs):
    sender.send_task(train_and_publish_task.name)
    sender.send_task(update_dictionaries_task.name)
