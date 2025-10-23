import json
import logging
import os
from collections.abc import Generator
from typing import List, Optional, Tuple

import httpx

log = logging.getLogger("cs2_etl")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def _generate_game_raw(
    path_to_games_raw_dir: str,
) -> Generator[Tuple[str, dict], None, None]:
    log.info(f"🔍 Сканирование директории с играми: {path_to_games_raw_dir}")
    for root, _, files in os.walk(path_to_games_raw_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                log.info(f"📂 Успешно загружен файл игры: {file_path}")
                yield file_path, data
            except Exception as e:
                log.warning(f"⚠️ Не удалось загрузить файл {file_path}: {e}")


# ----------------------------
# Entity Extractors
# ----------------------------


def _extract_map(game: dict) -> Optional[dict]:
    try:
        return {"map_id": game["map"]["id"], "name": game["map"]["name"]}
    except Exception as e:
        log.warning(f"⚠️ Не удалось получить данные карты: {e}")
        return None


def _extract_teams(game: dict) -> List[dict]:
    teams_dict = {}
    for p in game.get("players", []):
        team = p.get("team")
        if team and team.get("id") and team.get("name"):
            teams_dict[team["id"]] = {"team_id": team["id"], "name": team["name"]}
    return list(teams_dict.values())


def _extract_players(game: dict) -> List[dict]:
    players_dict = {}
    for p in game.get("players", []):
        player = p.get("player")
        if player and player.get("id") and player.get("name"):
            players_dict[player["id"]] = {
                "player_id": player["id"],
                "name": player["name"],
            }
    return list(players_dict.values())


# ----------------------------
# HTTP Sender
# ----------------------------


def _send_data(client: httpx.Client, endpoint: str, data: List[dict], base_url: str):
    if not data:
        return
    response = client.post(f"{base_url}/{endpoint}/save", json=data)
    response.raise_for_status()
    log.info(f"✅ Успешно отправлено на {endpoint}: {len(data)} записей")


def load_cs2_data(
    path_to_games_raw_dir: str,
    base_url: str,
    client: httpx.Client,
) -> dict[str, int]:
    log.info("🚀 Запуск процесса загрузки данных CS2 через HTTP API (sync)")

    total, success, error = 0, 0, 0
    client_provided = client is not None
    client = client or httpx.Client()

    try:
        for file_path, game in _generate_game_raw(path_to_games_raw_dir):
            total += 1
            log.info(f"⚙️ Обработка файла игры #{total}: {file_path}")
            try:
                map_data = _extract_map(game)
                teams = _extract_teams(game)
                players = _extract_players(game)

                _send_data(client, "maps", [map_data] if map_data else [], base_url)
                _send_data(client, "teams", teams, base_url)
                _send_data(client, "players", players, base_url)

                success += 1
                log.info(f"✅ Игра #{total} успешно загружена через API")
            except Exception as e:
                error += 1
                log.error(
                    f"❌ Ошибка при обработке файла {file_path}: {e}", exc_info=True
                )
    finally:
        if not client_provided:
            client.close()

    log.info("📊 Загрузка завершена")
    log.info(f"Всего файлов: {total}, Успешно: {success}, Ошибки: {error}")

    return {"total": total, "success": success, "error": error}
