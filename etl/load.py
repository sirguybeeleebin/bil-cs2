import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("etl_worker")


def _save_json(data: Any, dir_path: Path, filename: str) -> None:
    file_path = dir_path / f"{filename}.json"
    dir_path.mkdir(parents=True, exist_ok=True)
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.error(f"Не удалось сохранить файл {file_path}: {e}")


def load_map(map_data: dict, map_dir: Path) -> None:
    if not map_data:
        return
    map_id = map_data.get("map_id", "unknown_map")
    _save_json(map_data, map_dir, map_id)


def load_teams(teams_data: list[dict], team_dir: Path) -> None:
    if not teams_data:
        return
    for team in teams_data:
        team_id = team.get("team_id", "unknown_team")
        _save_json(team, team_dir, team_id)


def load_players(players_data: list[dict], player_dir: Path) -> None:
    if not players_data:
        return
    for player in players_data:
        player_id = player.get("player_id", "unknown_player")
        _save_json(player, player_dir, player_id)


def load_game_flatten(game_rows: list[dict], game_flatten_dir: Path) -> None:
    if not game_rows:
        return
    game_id = game_rows[0].get("game_id", "unknown_game")
    _save_json(game_rows, game_flatten_dir, game_id)
