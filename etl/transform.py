import logging
from collections import defaultdict

from dateutil.parser import parse

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


def validate_game(game: dict) -> bool:
    try:
        int(game["map"]["id"])
        parse(game["begin_at"])
        team_players = defaultdict(list)
        for p in game["players"]:
            team_players[p["team"]["id"]].append(p["player"]["id"])
        if len(team_players) != 2:
            return False
        for t_id, p_ids in team_players.items():
            if len(set(p_ids)) != 5:
                return False
        t1_id, t2_id = list(team_players.keys())
        rounds = []
        for r in game["rounds"]:
            if r["round"] is None:
                continue
            if r["ct"] not in (t1_id, t2_id):
                return False
            if r["terrorists"] not in (t1_id, t2_id):
                return False
            if r["winner_team"] not in (t1_id, t2_id):
                return False
            rounds.append(r["round"])
        if min(rounds) != 1:
            return False
        if max(rounds) < 16:
            return False
        return True
    except Exception:
        return None


def transform_game_flatten(game: dict) -> list[dict]:
    data = {}
    data["game_id"] = int(game["id"])
    data["begin_at"] = parse(game["begin_at"]).isoformat()
    data["map_id"] = int(game["map"]["id"])

    team_players = defaultdict(list)
    team_pair = {}
    player_stats = {}
    for p in game["players"]:
        team_players[p["team"]["id"]].append(p["player"]["id"])
        player_stats[p["player"]["id"]] = {
            "kills": p.get("kills", 0) or 0,
            "deaths": p.get("deaths", 0) or 0,
            "assists": p.get("assists", 0) or 0,
            "headshots": p.get("headshots", 0) or 0,
            "flash_assists": p.get("flash_assists", 0) or 0,
            "first_kills_diff": p.get("first_kills_diff", 0) or 0,
            "k_d_diff": p.get("k_d_diff", 0) or 0,
            "adr": p.get("adr", 0.0) or 0.0,
            "kast": p.get("kast", 0) or 0.0,
            "rating": p.get("rating", 0) or 0.0,
        }
    t1_id, t2_id = sorted(list(team_players.keys()))
    team_pair[t1_id] = t2_id
    team_pair[t2_id] = t1_id

    rows = []
    for t_id, p_ids in team_players.items():
        t_opp_id = team_pair[t_id]
        p_opp_ids = team_players[t_opp_id]
        for p_id in p_ids:
            for p_opp_id in p_opp_ids:
                for r in game["rounds"]:
                    row = {}
                    row.update(data)
                    row["team_id"] = t_id
                    row["team_opponent_id"] = t_opp_id
                    row["player_id"] = p_id
                    row["player_opponent_id"] = p_opp_id
                    row["round_id"] = r["round"]
                    row["round_is_ct"] = int(r["ct"] == t_id)
                    row["round_outcome"] = {
                        "eliminated": 1,
                        "defused": 2,
                        "exploded": 3,
                        4: "timeout",
                    }.get(r.get("outcome"), 0)
                    row["round_win"] = int(r["winner_team"] == t_id)
                    row.update(player_stats[p_id])
                    rows.append(row)
    return rows
