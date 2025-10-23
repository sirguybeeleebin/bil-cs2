import argparse
import asyncio
import logging
import os
from etl.etl import load_cs2_data
from dotenv import load_dotenv
import httpx

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("cs2_main")


def parse_config() -> dict:
    """
    Парсит CLI аргументы и загружает .env.
    Возвращает конфигурацию.
    """
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
        "env_file": args.env_file,
    }
    return config


async def load_cs2_games_periodically(config: dict, client: httpx.Client):
    """
    config: словарь с настройками
    client: httpx.Client (можно повторно использовать)
    """
    while True:
        try:
            log.info("⚙️ Запуск загрузки CS2 данных...")
            result = load_cs2_data(config["games_dir"], config["base_url"], client)
            log.info(f"✅ Задача завершена: {result}")
        except Exception as e:
            log.error(f"❌ Ошибка при загрузке CS2 данных: {e}", exc_info=True)

        log.info(f"⏱ Ждем {config['load_interval']} секунд до следующей загрузки...")
        await asyncio.sleep(config["load_interval"])


async def main():
    config = parse_config()
    log.info(f"✅ Загружен .env файл: {config['env_file']}")
    log.info(f"⏱ Интервал загрузки: {config['load_interval']} секунд")
    
    async with httpx.AsyncClient() as client:
        log.info("🚀 CS2 Loader стартует...")
        await load_cs2_games_periodically(config, client)


if __name__ == "__main__":
    asyncio.run(main())
