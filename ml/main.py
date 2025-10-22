import json
import logging
import os
from uuid import uuid4

import joblib
import redis
from celery import Celery
from celery.schedules import crontab
from celery.signals import worker_ready
from dotenv import load_dotenv

from ml.train_model import train_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("ml_main")

load_dotenv()
log.info("✅ Загружен .env файл")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

GAMES_RAW_DIR = os.getenv("PATH_TO_GAMES_RAW_DIR", "data/games_raw")
ML_RESULTS_DIR = os.getenv("PATH_TO_ML_RESULTS_DIR", "data/ml_results")

redis_uri = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
redis_client = redis.Redis.from_url(redis_uri)
try:
    redis_client.ping()
    log.info(f"✅ Подключено к Redis: {REDIS_HOST}:{REDIS_PORT} (db={REDIS_DB})")
except Exception as e:
    log.error(f"❌ Ошибка подключения к Redis: {e}")
    raise

celery_app = Celery("ml_worker", broker=redis_uri, backend=redis_uri)

@celery_app.task(name="ml_worker.train_model_task")
def train_model_task(path_to_games_raw_dir: str, path_to_ml_results_dir: str):
    task_id = uuid4()
    log.info(f"⚙️ Запуск обучения модели, task_id={task_id}...")

    predictor, metrics = train_model(path_to_games_raw_dir)

    os.makedirs(path_to_ml_results_dir, exist_ok=True)

    model_path = os.path.join(path_to_ml_results_dir, f"{task_id}.pkl")
    joblib.dump(predictor.pipeline, model_path)

    metrics_path = os.path.join(path_to_ml_results_dir, f"{task_id}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    redis_client.publish(
        "ml_updates",
        json.dumps(
            {
                "event": "model_trained",
                "task_id": str(task_id),
                "model_path": model_path,
                "metrics_path": metrics_path,
            }
        ),
    )

    log.info(f"✅ Задача обучения модели завершена, task_id={task_id}")
    return f"ML training finished, task_id={task_id}"

@worker_ready.connect
def at_start(**kwargs):
    log.info("🚀 Воркер ML готов — триггерим первую задачу обучения модели...")
    train_model_task.delay(GAMES_RAW_DIR, ML_RESULTS_DIR)

celery_app.conf.beat_schedule = {
    "daily-train-model": {
        "task": "ml_worker.train_model_task",
        "schedule": crontab(minute=0, hour=0),
        "args": (GAMES_RAW_DIR, ML_RESULTS_DIR),
    },
}
celery_app.conf.task_routes = {"ml_worker.*": {"queue": "ml"}}

log.info("✅ Worker и Beat настроены и готовы к работе.")
