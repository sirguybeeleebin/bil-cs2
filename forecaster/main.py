import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import AsyncGenerator, Optional

import asyncpg
import joblib
import redis.asyncio as redis
from dotenv import load_dotenv
from fastapi import FastAPI

from forecaster.repositories.ml_results import MLResultsRepository

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("cs2_forecaster")


def get_config() -> dict:
    parser = argparse.ArgumentParser(description="CS2 app config")
    parser.add_argument(
        "--env_file",
        type=str,
        default=".env",
        help="Path to .env file (default: .env)",
    )
    args = parser.parse_args()
    load_dotenv(args.env_file)

    return {
        "redis_host": os.getenv("REDIS_HOST", "localhost"),
        "redis_port": int(os.getenv("REDIS_PORT", 6379)),
        "redis_db": int(os.getenv("REDIS_DB", 0)),
        "redis_channel": os.getenv("REDIS_CHANNEL", "ml_models"),
        "postgres_user": os.getenv("POSTGRES_USER", "user"),
        "postgres_password": os.getenv("POSTGRES_PASSWORD", "password"),
        "postgres_host": os.getenv("POSTGRES_HOST", "localhost"),
        "postgres_port": int(os.getenv("POSTGRES_PORT", 5432)),
        "postgres_db": os.getenv("POSTGRES_DB", "db"),
        "postgres_min_size": int(os.getenv("POSTGRES_MIN_SIZE", 1)),
        "postgres_pool_size": int(os.getenv("POSTGRES_POOL_SIZE", 10)),
        "postgres_max_idle_cons": int(os.getenv("POSTGRES_MAX_IDLE_CONS", 5)),
    }


async def consume_models(
    redis_client: redis.Redis,
    channel_name: str,
    ml_repo: MLResultsRepository,
    app: FastAPI,
):
    pubsub = redis_client.pubsub()
    await pubsub.subscribe(channel_name)
    log.info(f"🎧 Подключено к Redis, канал: {channel_name}")

    async for message in pubsub.listen():
        if message is None or message["type"] != "message":
            continue
        try:
            payload = json.loads(message["data"])
            predictor_path = payload["predictor_path"]
            metrics_path = payload["metrics_path"]
            task_id = payload.get("task_id", "unknown")

            log.info(f"📥 Получено новое сообщение: task_id={task_id}")
            log.info(f"   Predictor path: {predictor_path}")
            log.info(f"   Metrics path: {metrics_path}")

            await ml_repo.upsert(task_id, predictor_path, metrics_path)

            path = Path(predictor_path)
            if path.exists():
                predictor = joblib.load(path)
                app.state.predictor = predictor
                log.info(
                    f"✅ Active ML predictor updated in memory from: {predictor_path}"
                )
            else:
                log.error(f"❌ Predictor file not found: {predictor_path}")

        except Exception as e:
            log.error(f"❌ Ошибка при обработке сообщения: {e}", exc_info=True)


async def lifespan(app: FastAPI) -> AsyncGenerator:
    config = get_config()

    redis_client = redis.Redis(
        host=config["redis_host"], port=config["redis_port"], db=config["redis_db"]
    )

    pg_pool = await asyncpg.create_pool(
        user=config["postgres_user"],
        password=config["postgres_password"],
        database=config["postgres_db"],
        host=config["postgres_host"],
        port=config["postgres_port"],
        min_size=config["postgres_min_size"],
        max_size=config["postgres_pool_size"],
        max_inactive_connection_lifetime=config["postgres_max_idle_cons"],
    )
    log.info("✅ PostgreSQL pool создан")

    ml_repo = MLResultsRepository(pg_pool)

    predictor: Optional[object] = None
    latest_path = await ml_repo.get_latest_predictor_path()
    if latest_path:
        try:
            predictor = joblib.load(latest_path)
            log.info(f"✅ Loaded active ML predictor from: {latest_path}")
        except Exception as e:
            log.error(
                f"❌ Failed to load ML predictor from {latest_path}: {e}", exc_info=True
            )
    else:
        log.warning("⚠️ No ML predictor found in database at startup")

    app.state.predictor = predictor
    app.state.ml_repo = ml_repo

    consumer_task = asyncio.create_task(
        consume_models(redis_client, config["redis_channel"], ml_repo, app)
    )
    log.info("🚀 Redis consumer запущен")

    try:
        yield
    finally:
        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            log.info("🔹 Consumer task отменен")
        await redis_client.close()
        await redis_client.connection_pool.disconnect()
        await pg_pool.close()
        log.info("🔌 Redis и PostgreSQL соединения закрыты")


app = FastAPI(lifespan=lifespan)
