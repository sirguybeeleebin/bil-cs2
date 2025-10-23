import argparse
import asyncio
import logging
import os
from datetime import datetime

from dotenv import load_dotenv
from etl.etl import load_cs2_data

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("cs2_main")


def parse_args():
    parser = argparse.ArgumentParser(description="CS2 periodic data loader")
    parser.add_argument(
        "--env_file",
        type=str,
        default=".env",
        help="Path to .env file (default: .env)",
    )
    return parser.parse_args()


async def load_cs2_games_periodically(games_dir: str, base_url: str, interval: int):
    while True:
        try:
            log.info("⚙️ Запуск загрузки CS2 данных...")
            result = load_cs2_data(games_dir, base_url)
            log.info(f"✅ Задача завершена: {result}")
        except Exception as e:
            log.error(f"❌ Ошибка при загрузке CS2 данных: {e}", exc_info=True)
        log.info(f"⏱ Ждем {interval} секунд до следующей загрузки...")
        await asyncio.sleep(interval)


if __name__ == "__main__":
    args = parse_args()
    load_dotenv(args.env_file)
    log.info(f"✅ Загружен .env файл: {args.env_file}")

    GAMES_RAW_DIR = os.getenv("PATH_TO_GAMES_RAW_DIR", "data/games_raw")
    BASE_URL = os.getenv("CS2_API_BASE_URL", "http://localhost:8000")
    LOAD_INTERVAL = int(os.getenv("LOAD_INTERVAL_SECONDS", 60 * 60))

    log.info("🚀 Worker CS2 стартует...")
    asyncio.run(load_cs2_games_periodically(GAMES_RAW_DIR, BASE_URL, LOAD_INTERVAL))
