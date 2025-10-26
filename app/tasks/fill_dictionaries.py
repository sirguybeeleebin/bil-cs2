import json
import logging
import os

from celery import shared_task

from app.repositories.map import MapRepository
from app.repositories.player import PlayerRepository
from app.repositories.team import TeamRepository

log = logging.getLogger(__name__)


def _get_map_dict(data: dict) -> dict:
    try:
        map_data = {"map_id": data["map"]["id"], "name": data["map"]["name"]}
        return map_data
    except (KeyError, TypeError) as e:
        log.error(f"Ошибка при загрузке карты: {e}")
        return {}


def _get_teams_dicts(data: dict) -> list[dict]:
    teams = []
    for p in data.get("players", []):
        try:
            team_data = {"team_id": p["team"]["id"], "name": p["team"]["name"]}
            teams.append(team_data)
        except (KeyError, TypeError) as e:
            log.error(f"Ошибка при загрузке команды: {e}")
    return teams


def _get_players_dicts(data: dict) -> list[dict]:
    players = []
    for p in data.get("players", []):
        try:
            player_data = {"player_id": p["player"]["id"], "name": p["player"]["name"]}
            players.append(player_data)
        except (KeyError, TypeError) as e:
            log.error(f"Ошибка при загрузке игрока: {e}")
    return players


def make_fill_dictionaries_task(
    map_repository: MapRepository,
    team_repository: TeamRepository,
    player_repository: PlayerRepository,
):
    @shared_task
    def fill_dictionaries_task(path_to_games_raw_dir: str):
        if not os.path.exists(path_to_games_raw_dir):
            log.error(f"Директория {path_to_games_raw_dir} не существует")
            return {
                "status": "ошибка",
                "message": f"{path_to_games_raw_dir} не существует",
            }

        log.info(f"Начало загрузки данных из {path_to_games_raw_dir}")
        filenames = os.listdir(path_to_games_raw_dir)
        total = len(filenames)

        for idx, filename in enumerate(filenames):
            log.info(f"Обработка файла {idx + 1}/{total}: {filename}")
            file_path = os.path.join(path_to_games_raw_dir, filename)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                map_data = _get_map_dict(data)
                if map_data:
                    result = map_repository.upsert(map_data)
                    if not result:
                        log.error(f"Не удалось создать или обновить карту: {map_data}")

                teams = _get_teams_dicts(data)
                for team in teams:
                    result = team_repository.upsert(team)
                    if not result:
                        log.error(f"Не удалось создать или обновить команду: {team}")

                players = _get_players_dicts(data)
                for player in players:
                    result = player_repository.upsert(player)
                    if not result:
                        log.error(f"Не удалось создать или обновить игрока: {player}")

            except (json.JSONDecodeError, FileNotFoundError, TypeError, KeyError) as e:
                log.error(f"Ошибка при обработке файла {file_path}: {e}")
                continue

        log.info("Загрузка данных завершена")
        return {"status": "готово"}

    return fill_dictionaries_task
