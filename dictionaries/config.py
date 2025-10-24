import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR: Path = Path(os.getenv("DATA_DIR", "../data"))
MAP_DIR: Path = BASE_DIR / "maps"
TEAM_DIR: Path = BASE_DIR / "teams"
PLAYER_DIR: Path = BASE_DIR / "players"
GAME_FLATTEN_DIR: Path = BASE_DIR / "games_flatten"

for d in [MAP_DIR, TEAM_DIR, PLAYER_DIR, GAME_FLATTEN_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Auth service URL
AUTH_SERVICE_URL: str = os.getenv(
    "AUTH_SERVICE_URL", "http://auth-service/service/token/me"
)
