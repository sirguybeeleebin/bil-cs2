import logging
import os

import psycopg2
from celery import Celery
from celery.schedules import crontab
from celery.signals import worker_ready
from dotenv import load_dotenv

from etl.etl import load_cs2_data_to_postgres

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("cs2_main")

load_dotenv()
log.info("✅ Загружен .env файл")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
GAMES_RAW_DIR = os.getenv("PATH_TO_GAMES_RAW_DIR", "data/games_raw")

PG_CONN_PARAMS = {
    "user": os.getenv("POSTGRES_USER", "cs2_user"),
    "password": os.getenv("POSTGRES_PASSWORD", "cs2_password"),
    "dbname": os.getenv("POSTGRES_DB", "cs2_db"),
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", 5432)),
}

redis_uri = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
celery_app = Celery("cs2_worker", broker=redis_uri, backend=redis_uri)

@celery_app.task(name="cs2_worker.load_cs2_games_task")
def load_cs2_games_task():
    log.info("⚙️ Запуск задачи загрузки CS2 данных в PostgreSQL...")
    conn = psycopg2.connect(**PG_CONN_PARAMS)
    try:
        result = load_cs2_data_to_postgres(GAMES_RAW_DIR, conn)
        log.info(f"✅ Задача завершена: {result}")
        return result
    finally:
        conn.close()
        log.info("🔒 Соединение с PostgreSQL закрыто")

@worker_ready.connect
def at_start(**kwargs):
    log.info("🚀 Воркер CS2 готов — триггерим первую задачу...")
    load_cs2_games_task.delay()

celery_app.conf.beat_schedule = {
    "hourly-load-cs2-games": {
        "task": "cs2_worker.load_cs2_games_task",
        "schedule": crontab(minute=0),
        "args": (),
    },
}
celery_app.conf.task_routes = {"cs2_worker.*": {"queue": "cs2"}}

log.info("✅ Worker и Beat настроены и готовы к работе.")
