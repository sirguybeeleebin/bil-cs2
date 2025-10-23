import argparse
import asyncio
import logging
import os
from etl.etl import load_cs2_data
from dotenv import load_dotenv

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
        "base_url": os.getenv("CS2_API_BASE_URL", "http://localhost:8000"),
        "load_interval": int(os.getenv("LOAD_INTERVAL_SECONDS", 60 * 60)),        
    }
    return config


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
    config = parse_config()
    log.info(f"✅ Загружен .env файл: {config['env_file']}")
    log.info("🚀 Worker CS2 стартует...")
    asyncio.run(
        load_cs2_games_periodically(
            config["games_dir"], config["base_url"], config["load_interval"]
        )
    )
