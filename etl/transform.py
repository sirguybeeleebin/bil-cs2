import logging

logger = logging.getLogger("etl_worker")


def transform_map(game: dict) -> dict | None:
    try:
        return {"map_id": str(game["map"]["id"]), "name": game["map"]["name"] or ""}
    except KeyError as e:
        logger.error(f"Отсутствует ключ карты в данных игры: {e}")
        return None


def transform_team(game: dict) -> list[dict] | None:
    try:
        teams: list[dict] = []
        for p in game["players"]:
            team: dict = {
                "team_id": str(p["team"]["id"]),
                "name": p["team"]["name"] or "",
            }
            if team not in teams:
                teams.append(team)
        return teams
    except KeyError as e:
        logger.error(f"Отсутствует ключ команды в данных игры: {e}")
        return None


def transform_player(game: dict) -> list[dict] | None:
    try:
        players: list[dict] = []
        for p in game["players"]:
            player: dict = {
                "player_id": str(p["player"]["id"]),
                "name": p["player"]["name"] or "",
            }
            if player not in players:
                players.append(player)
        return players
    except KeyError as e:
        logger.error(f"Отсутствует ключ игрока в данных игры: {e}")
        return None
