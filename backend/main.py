import pickle
import argparse
from fastapi import FastAPI, APIRouter
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import uvicorn
import aiosqlite
from contextlib import asynccontextmanager

from backend.repositories import TeamRepository, PlayerRepository
from backend.services import TeamService, PlayerService, ForecastService
from backend.routers.team import create_team_router
from backend.routers.player import create_player_router
from backend.routers.forecast import create_forecast_router

class Settings(BaseSettings):
    model_path: str = Field(default="data/ml_results/8eaa28297645dca5.pickle")
    sqlite_url: str = Field(default="../sqlite/db.sqlite")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    debug: bool = Field(default=False)
    api_prefix: str = Field(default="/api/v1")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

def get_settings() -> Settings:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_file", type=str)
    args = parser.parse_args()
    if args.env_file:
        return Settings(_env_file=args.env_file)
    return Settings()

def create_app(settings: Settings) -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        with open(settings.model_path, "rb") as f:
            ml_model = pickle.load(f)

        conn = await aiosqlite.connect(settings.sqlite_url)

        team_service = TeamService(TeamRepository(conn))
        player_service = PlayerService(PlayerRepository(conn))
        forecast_service = ForecastService(ml_model, team_service, player_service)

        app.state.conn = conn
        app.state.team_service = team_service
        app.state.player_service = player_service
        app.state.forecast_service = forecast_service

        yield

        await conn.close()

    app = FastAPI(lifespan=lifespan)
    
    router = APIRouter(prefix=settings.api_prefix)
    router.include_router(create_team_router(app.state.team_service))
    router.include_router(create_player_router(app.state.player_service))
    router.include_router(create_forecast_router(app.state.forecast_service))

    app.include_router(router)
    return app

def run_uvicorn(app: FastAPI, settings: Settings):
    uvicorn.run(app, host=settings.host, port=settings.port, reload=settings.debug)

def main():
    settings = get_settings()
    app = create_app(settings)
    run_uvicorn(app, settings)

if __name__ == "__main__":
    main()
