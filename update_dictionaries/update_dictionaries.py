import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from dateutil.parser import parse

log = logging.getLogger(__name__)


def _validate_game(game: dict[str, Any]) -> bool:
    try:
        parse(game["begin_at"])
        int(game["map"]["id"])
        team_players: dict[Any, list[Any]] = defaultdict(list)
        for p in game["players"]:
            team_players[p["team"]["id"]].append(p["player"]["id"])
        if len(team_players) != 2:
            log.debug(f"Игра {game['id']} имеет не 2 команды")
            return False
        for _, p_ids in team_players.items():
            if len(set(p_ids)) != 5:
                log.debug(
                    f"Игра {game['id']} имеет некорректное количество игроков в команде"
                )
                return False
        t1_id, t2_id = list(team_players.keys())
        rounds: list[int] = []
        for r in game["rounds"]:
            if r["round"] is None:
                continue
            if r["winner_team"] not in (t1_id, t2_id):
                log.debug(
                    f"Игра {game['id']} содержит некорректного победителя в раунде"
                )
                return False
            rounds.append(r["round"])
        if min(rounds, default=1) != 1 or max(rounds, default=0) < 16:
            log.debug(f"Игра {game['id']} не соответствует диапазону раундов")
            return False
        return True
    except Exception as e:
        log.warning(f"Ошибка при валидации игры {game.get('id', 'неизвестно')}: {e}")
        return False


def update_dictionaries(
    games_raw_dir: str, maps_dir: str, teams_dir: str, players_dir: str
):
    for directory in [maps_dir, teams_dir, players_dir]:
        os.makedirs(directory, exist_ok=True)

    now_iso = datetime.now(timezone.utc).isoformat()
    log.info(f"Начало обработки файлов в папке: {games_raw_dir}")

    for filename in os.listdir(games_raw_dir):
        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(games_raw_dir, filename)
        log.info(f"Чтение файла: {filename}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            log.error(e)
            continue

        if not _validate_game(data):
            log.info(f"Файл {filename} пропущен, игра не прошла валидацию")
            continue

        map_data = {
            "map_id": data["map"]["id"],
            "name": data["map"]["name"],
            "updated_at": now_iso,
        }
        map_file = os.path.join(maps_dir, f"{map_data['map_id']}.json")
        with open(map_file, "w", encoding="utf-8") as f:
            json.dump(map_data, f, ensure_ascii=False, indent=2)
        log.info(f"Сохранили карту: {map_data['name']} (ID: {map_data['map_id']})")

        for p in data["players"]:
            team_data = {
                "team_id": p["team"]["id"],
                "name": p["team"]["name"],
                "updated_at": now_iso,
            }
            team_file = os.path.join(teams_dir, f"{team_data['team_id']}.json")
            with open(team_file, "w", encoding="utf-8") as f:
                json.dump(team_data, f, ensure_ascii=False, indent=2)
            log.info(
                f"Сохранили команду: {team_data['name']} (ID: {team_data['team_id']})"
            )

        for p in data["players"]:
            player_data = {
                "player_id": p["player"]["id"],
                "name": p["player"]["name"],
                "updated_at": now_iso,
            }
            player_file = os.path.join(players_dir, f"{player_data['player_id']}.json")
            with open(player_file, "w", encoding="utf-8") as f:
                json.dump(player_data, f, ensure_ascii=False, indent=2)
            log.info(
                f"Сохранили игрока: {player_data['name']} (ID: {player_data['player_id']})"
            )

    log.info("Обработка завершена.")
