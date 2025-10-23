import json
import logging
import os

import joblib
from celery import Celery
from celery.schedules import crontab
from celery.signals import worker_ready
from dotenv import load_dotenv

from ml.train_model import train_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("cs2_ml_worker")

load_dotenv()
log.info("✅ Загружен .env файл")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
GAMES_RAW_DIR = os.getenv("PATH_TO_GAMES_RAW_DIR", "data/games_raw")
ML_RESULTS_DIR = os.getenv("PATH_TO_ML_RESULTS_DIR", "data/ml_results")

os.makedirs(ML_RESULTS_DIR, exist_ok=True)

redis_uri = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
celery_app = Celery("cs2_ml_worker", broker=redis_uri, backend=redis_uri)


@celery_app.task(name="cs2_ml_worker.train_model_task", bind=True)
def train_model_task(self):
    log.info("⚙️ Запуск задачи обучения модели CS2...")
    task_id = self.request.id
    try:
        predictor, metrics = train_model(GAMES_RAW_DIR)

        predictor_path = os.path.join(ML_RESULTS_DIR, f"predictor_{task_id}.joblib")
        metrics_path = os.path.join(ML_RESULTS_DIR, f"metrics_{task_id}.json")

        joblib.dump(predictor, predictor_path)
        log.info(f"✅ Predictor сохранен: {predictor_path}")

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)
        log.info(f"✅ Метрики сохранены: {metrics_path}")

        return {"predictor_path": predictor_path, "metrics_path": metrics_path}

    except Exception as e:
        log.error(f"❌ Ошибка при обучении модели: {e}", exc_info=True)
        raise e


@worker_ready.connect
def at_start(**kwargs):
    log.info("🚀 Воркер CS2 ML готов — триггерим задачу обучения модели...")
    train_model_task.delay()


# Optional: schedule periodic model training (daily at midnight)
celery_app.conf.beat_schedule = {
    "daily-train-model": {
        "task": "cs2_ml_worker.train_model_task",
        "schedule": crontab(hour=0, minute=0),
        "args": (),
    },
}

celery_app.conf.task_routes = {"cs2_ml_worker.*": {"queue": "cs2_ml"}}

log.info("✅ ML Worker и Beat настроены и готовы к работе.")
