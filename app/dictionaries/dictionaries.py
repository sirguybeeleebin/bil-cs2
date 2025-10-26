import logging

log = logging.getLogger(__name__)


def get_map_dict(data: dict) -> dict:
    try:
        map_data = {"map_id": data["map"]["id"], "name": data["map"]["name"]}
        return map_data
    except (KeyError, TypeError) as e:
        log.error(f"Ошибка при загрузке карты: {e}")
        return {}


def get_teams_dicts(data: dict) -> list[dict]:
    teams = []
    for p in data.get("players", []):
        try:
            team_data = {"team_id": p["team"]["id"], "name": p["team"]["name"]}
            teams.append(team_data)
        except (KeyError, TypeError) as e:
            log.error(f"Ошибка при загрузке команды: {e}")
    return teams


def get_players_dicts(data: dict) -> list[dict]:
    players = []
    for p in data.get("players", []):
        try:
            player_data = {"player_id": p["player"]["id"], "name": p["player"]["name"]}
            players.append(player_data)
        except (KeyError, TypeError) as e:
            log.error(f"Ошибка при загрузке игрока: {e}")
    return players
