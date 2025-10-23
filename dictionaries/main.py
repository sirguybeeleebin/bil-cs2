import argparse
import logging

import asyncpg
import uvicorn
from fastapi import APIRouter, FastAPI
from pydantic_settings import BaseSettings, SettingsConfigDict
from repositories.map import make_map_repository
from repositories.player import make_player_repository
from repositories.team import make_team_repository
from routers.map import make_map_router
from routers.player import make_player_router
from routers.team import make_team_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    TITLE: str = "Сервис игровой словарной базы"
    VERSION: str = "/api/v1"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000

    POSTGRES_DSN: str = "postgresql://cs2_dictionaries_user:cs2_dictionaries_password@localhost:5432/cs2_dictionaries_db"
    POSTGRES_MIN_SIZE: int = 1
    POSTGRES_MAX_SIZE: int = 10
    POSTGRES_IDLE_TIMEOUT: float = 60.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


parser = argparse.ArgumentParser(description="Запуск сервиса игровой словарной базы")
parser.add_argument("--env_file", type=str, default=".env", help="Путь к файлу .env")
args = parser.parse_args()

logger.info(f"Используется файл окружения: {args.env_file}")
settings = Settings(_env_file=args.env_file)


async def lifespan(app: FastAPI):
    logger.info("Подключение к PostgreSQL...")

    pool = await asyncpg.create_pool(
        dsn=settings.POSTGRES_DSN,
        min_size=settings.POSTGRES_MIN_SIZE,
        max_size=settings.POSTGRES_MAX_SIZE,
        idle_timeout=settings.POSTGRES_IDLE_TIMEOUT,
    )

    map_repo = make_map_repository(pool)
    team_repo = make_team_repository(pool)
    player_repo = make_player_repository(pool)

    map_router = make_map_router(map_repo)
    team_router = make_team_router(team_repo)
    player_router = make_player_router(player_repo)

    router = APIRouter(prefix=settings.VERSION)
    router.include_router(map_router)
    router.include_router(team_router)
    router.include_router(player_router)

    app.include_router(router)
    logger.info("✅ Сервис запущен.")

    yield

    logger.info("Закрытие подключения к PostgreSQL...")
    await pool.close()


app = FastAPI(title=settings.TITLE, lifespan=lifespan)


if __name__ == "__main__":
    logger.info(
        f"Запуск сервера на http://{settings.APP_HOST}:{settings.APP_PORT}{settings.VERSION}"
    )
    uvicorn.run(app, host=settings.APP_HOST, port=settings.APP_PORT, reload=True)
