from datetime import datetime

from pydantic import BaseModel


class MapData(BaseModel):
    map_id: int
    name: str | None = ""


class TeamData(BaseModel):
    team_id: int
    name: str | None = ""


class PlayerData(BaseModel):
    player_id: int
    name: str | None = ""


class GameFlattenData(BaseModel):
    game_id: int
    begin_at: datetime
    map_id: int
    team_id: int
    team_opponent_id: int
    player_id: int
    player_opponent_id: int
    round_id: int
    round_is_ct: int
    round_outcome: int
    round_win: int
    kills: int = 0
    deaths: int = 0
    assists: int = 0
    headshots: int = 0
    flash_assists: int = 0
    first_kills_diff: int = 0
    k_d_diff: int = 0
    adr: float = 0.0
    kast: float = 0.0
    rating: float = 0.0
