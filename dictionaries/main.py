from fastapi import FastAPI, Header, Response, status

from dictionaries.auth import lifespan, validate_token
from dictionaries.config import GAME_FLATTEN_DIR, MAP_DIR, PLAYER_DIR, TEAM_DIR
from dictionaries.models import GameFlattenData, MapData, PlayerData, TeamData
from dictionaries.storage import save_json

app = FastAPI(title="Dictionaries ETL", lifespan=lifespan)


@app.post("/maps/", status_code=status.HTTP_200_OK)
async def post_map(
    map_data: MapData, authorization: str | None = Header(None)
) -> Response:
    await validate_token(authorization, app)
    save_json(MAP_DIR / f"{map_data.map_id}.json", map_data.model_dump())
    return Response(status_code=status.HTTP_200_OK)


@app.post("/teams/", status_code=status.HTTP_200_OK)
async def post_team(
    team_data: TeamData, authorization: str | None = Header(None)
) -> Response:
    await validate_token(authorization, app)
    save_json(TEAM_DIR / f"{team_data.team_id}.json", team_data.model_dump())
    return Response(status_code=status.HTTP_200_OK)


@app.post("/players/", status_code=status.HTTP_200_OK)
async def post_player(
    player_data: PlayerData, authorization: str | None = Header(None)
) -> Response:
    await validate_token(authorization, app)
    save_json(PLAYER_DIR / f"{player_data.player_id}.json", player_data.model_dump())
    return Response(status_code=status.HTTP_200_OK)


@app.post("/games_flatten/", status_code=status.HTTP_200_OK)
async def post_game_flatten(
    game_data: GameFlattenData, authorization: str | None = Header(None)
) -> Response:
    await validate_token(authorization, app)
    save_json(GAME_FLATTEN_DIR / f"{game_data.game_id}.json", game_data.model_dump())
    return Response(status_code=status.HTTP_200_OK)
