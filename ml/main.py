import argparse
import asyncio
import json
import logging
import os
import uuid

import joblib
import redis.asyncio as redis
from dotenv import load_dotenv

from ml.train_model import train_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("cs2_ml_worker")


def parse_config() -> dict:
    parser = argparse.ArgumentParser(description="ML model periodic worker")
    parser.add_argument(
        "--env_file",
        type=str,
        default=".env",
        help="Path to .env file (default: .env)",
    )
    args = parser.parse_args()
    load_dotenv(args.env_file)

    config = {
        "games_dir": os.getenv("PATH_TO_GAMES_RAW_DIR", "data/games_raw"),
        "results_dir": os.getenv("PATH_TO_ML_RESULTS_DIR", "data/ml_results"),
        "train_interval": int(os.getenv("TRAIN_INTERVAL_SECONDS", 24 * 60 * 60)),
        "redis_host": os.getenv("REDIS_HOST", "localhost"),
        "redis_port": int(os.getenv("REDIS_PORT", 6379)),
        "redis_db": int(os.getenv("REDIS_DB", 0)),
        "redis_queue": os.getenv("REDIS_QUEUE", "ml_models"),
        "env_file": args.env_file,
    }
    os.makedirs(config["results_dir"], exist_ok=True)
    return config


async def train_model_once(config: dict, redis_conn: redis.Redis) -> dict:
    """
    Однократная тренировка модели и публикация результатов в Redis.
    Возвращает словарь с путями к predictor и метрикам.
    """
    task_id = str(uuid.uuid4())
    log.info("⚙️ Запуск однократной тренировки модели CS2...")
    result = {}

    try:
        predictor, metrics = train_model(config["games_dir"])

        predictor_path = os.path.join(config["results_dir"], f"{task_id}.joblib")
        metrics_path = os.path.join(config["results_dir"], f"{task_id}.json")

        joblib.dump(predictor, predictor_path)
        log.info(f"✅ Predictor сохранен: {predictor_path}")

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)
        log.info(f"✅ Метрики сохранены: {metrics_path}")

        message = json.dumps(
            {
                "predictor_path": predictor_path,
                "metrics_path": metrics_path,
                "task_id": task_id,
            }
        )
        await redis_conn.lpush(config["redis_queue"], message)
        log.info(
            f"📤 Сообщение опубликовано в Redis queue '{config['redis_queue']}': {message}"
        )

        result = {
            "task_id": task_id,
            "predictor_path": predictor_path,
            "metrics_path": metrics_path,
        }

    except Exception as e:
        log.error(f"❌ Ошибка при обучении модели: {e}", exc_info=True)

    return result


async def train_model_periodically(config: dict, redis_conn: redis.Redis):
    while True:
        await train_model_once(config, redis_conn)
        log.info(f"⏱ Ждем {config['train_interval']} секунд до следующей тренировки...")
        await asyncio.sleep(config["train_interval"])


async def main():
    config = parse_config()
    log.info(f"✅ Загружен .env файл: {config['env_file']}")
    log.info(f"⏱ Интервал тренировки: {config['train_interval']} секунд")

    redis_uri = (
        f"redis://{config['redis_host']}:{config['redis_port']}/{config['redis_db']}"
    )
    redis_conn = redis.from_url(redis_uri)

    try:
        log.info("🚀 ML Worker стартует...")
        await train_model_periodically(config, redis_conn)
    finally:
        await redis_conn.close()
        log.info("🔌 Redis connection closed.")


if __name__ == "__main__":
    asyncio.run(main())
