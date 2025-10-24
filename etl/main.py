import logging
import os
from pathlib import Path

from celery import Celery, signals
from dotenv import load_dotenv

from etl.extract import generate_game_raw
from etl.load import load_game_flatten, load_map, load_players, load_teams
from etl.transform import (
    transform_game_flatten,
    transform_map,
    transform_player,
    transform_team,
    validate_game,
)

load_dotenv()


BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
BACKEND_URL = os.getenv("CELERY_BACKEND_URL", "redis://redis:6379/1")
TIMEZONE = os.getenv("CELERY_TIMEZONE", "UTC")
GAMES_RAW_DIR_PATH = os.getenv("GAMES_RAW_DIR_PATH", "/app/data/games_raw")


BASE_DIR: Path = Path(os.getenv("DATA_DIR", "/app/data"))
MAP_DIR: Path = BASE_DIR / "maps"
TEAM_DIR: Path = BASE_DIR / "teams"
PLAYER_DIR: Path = BASE_DIR / "players"
GAME_FLATTEN_DIR: Path = BASE_DIR / "games_flatten"

for d in [MAP_DIR, TEAM_DIR, PLAYER_DIR, GAME_FLATTEN_DIR]:
    d.mkdir(parents=True, exist_ok=True)


logger = logging.getLogger("etl_worker")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


app = Celery("etl_worker", broker=BROKER_URL, backend=BACKEND_URL)
app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone=TIMEZONE,
    enable_utc=True,
)


@app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=5)
def etl_pipeline(self):
    logger.info(f"Запуск ETL для: {GAMES_RAW_DIR_PATH}")

    processed = 0
    for game in generate_game_raw(GAMES_RAW_DIR_PATH):
        if not validate_game(game):
            logger.warning(f"⏭ Пропущена игра id={game.get('id')}")
            continue

        processed += 1
        try:
            load_map(transform_map(game), MAP_DIR)
            load_teams(transform_team(game) or [], TEAM_DIR)
            load_players(transform_player(game) or [], PLAYER_DIR)
            load_game_flatten(transform_game_flatten(game), GAME_FLATTEN_DIR)
            logger.info(f"Игра {game.get('id')}: загружена")
        except Exception as e:
            logger.error(f"Ошибка обработки игры {game.get('id')}: {e}")
            continue

    logger.info(f"✅ ETL завершен | обработано игр: {processed}")
    return f"Готово: {processed} игр"


_etl_started = False


@signals.worker_ready.connect
def at_start(sender=None, **kwargs):
    """Запускаем ETL при старте воркера Celery один раз."""
    global _etl_started
    if not _etl_started:
        logger.info("Worker готов ✅ Запуск ETL...")
        etl_pipeline.delay()
        _etl_started = True
