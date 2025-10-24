import json
import logging
from pathlib import Path
from typing import Generator

logger = logging.getLogger("etl_worker")


def generate_game_raw(path_to_dir: str) -> Generator[dict, None, None]:
    path = Path(path_to_dir)
    for json_file in path.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                game = json.load(f)  # читаем один файл как словарь
                yield game
        except json.JSONDecodeError as e:
            logger.error(f"Не удалось декодировать JSON из файла {json_file.name}: {e}")
        except Exception as e:
            logger.error(f"Неожиданная ошибка при чтении файла {json_file.name}: {e}")
