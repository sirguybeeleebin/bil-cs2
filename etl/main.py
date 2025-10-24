import logging
import os

import httpx
from celery import Celery, signals
from dotenv import load_dotenv

from etl.extract import generate_game_raw
from etl.get_service_token import get_service_token
from etl.load import load_map, load_players, load_teams
from etl.transform import transform_map, transform_player, transform_team

load_dotenv()

BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
BACKEND_URL = os.getenv("CELERY_BACKEND_URL", "redis://localhost:6379/1")
MAP_URL = os.getenv("MAP_URL", "http://example.com/maps")
TEAM_URL = os.getenv("TEAM_URL", "http://example.com/teams")
PLAYER_URL = os.getenv("PLAYER_URL", "http://example.com/players")
GAMES_RAW_DIR_PATH = os.getenv("GAMES_RAW_DIR_PATH", "data/games_raw")
TIMEZONE = os.getenv("CELERY_TIMEZONE", "UTC")

SERVICE_ID = os.getenv("SERVICE_ID")
SERVICE_SECRET = os.getenv("SERVICE_SECRET")
AUTH_URL = os.getenv("AUTH_URL", "http://auth-service/service/token")

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


@app.task
def etl_pipeline():
    logger.info(f"Запуск ETL для директории: {GAMES_RAW_DIR_PATH}")
    with httpx.Client() as client:
        token = get_service_token(client, SERVICE_ID, SERVICE_SECRET, AUTH_URL)
        if not token:
            logger.error("Токен сервиса отсутствует, ETL прерван")
            return "ETL прерван: отсутствует токен сервиса"

        headers = {"Authorization": f"Bearer {token}"}

        for game in generate_game_raw(GAMES_RAW_DIR_PATH):
            map_data = transform_map(game)
            teams_data = transform_team(game) or []
            players_data = transform_player(game) or []

            load_map(map_data, client, MAP_URL, headers=headers)
            load_teams(teams_data, client, TEAM_URL, headers=headers)
            load_players(players_data, client, PLAYER_URL, headers=headers)

    logger.info(f"ETL завершен для директории: {GAMES_RAW_DIR_PATH}")
    return f"ETL завершен для директории: {GAMES_RAW_DIR_PATH}"


@signals.worker_ready.connect
def at_start(sender=None, **kwargs):
    logger.info("Рабочий процесс готов. Запуск ETL...")
    etl_pipeline.delay()
