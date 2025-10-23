# main.py
import argparse
import asyncio
import logging
import os

import asyncpg
from dotenv import load_dotenv

from etl.etl import load_cs2_data

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("cs2_main")


def parse_config() -> dict:
    parser = argparse.ArgumentParser(description="CS2 periodic data loader")
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
        "database_url": os.getenv(
            "POSTGRES_DATABASE_URL", "postgresql://user:pass@localhost:5432/cs2_db"
        ),
        "load_interval": int(os.getenv("LOAD_INTERVAL_SECONDS", 60 * 60)),
    }
    return config


async def load_cs2_games_periodically(games_dir: str, database_url: str, interval: int):
    # Создаём пул соединений
    pool = await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=5)

    try:
        while True:
            try:
                log.info("⚙️ Запуск загрузки CS2 данных...")
                result = await load_cs2_data(games_dir, pool)
                log.info(f"✅ Задача завершена: {result}")
            except Exception as e:
                log.error(f"❌ Ошибка при загрузке CS2 данных: {e}", exc_info=True)

            log.info(f"⏱ Ждем {interval} секунд до следующей загрузки...")
            await asyncio.sleep(interval)
    finally:
        await pool.close()
        log.info("🛑 Пул соединений PostgreSQL закрыт")


if __name__ == "__main__":
    config = parse_config()
    log.info("✅ Загружен .env файл")
    log.info("🚀 Worker CS2 стартует...")
    asyncio.run(
        load_cs2_games_periodically(
            config["games_dir"], config["database_url"], config["load_interval"]
        )
    )
