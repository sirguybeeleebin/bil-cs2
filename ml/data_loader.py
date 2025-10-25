from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Generator

import numpy as np
from dateutil.parser import parse


def generate_game_raw(path_to_dir: str) -> Generator[dict[str, Any], None, None]:
    path = Path(path_to_dir)
    for json_file in path.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                yield json.load(f)
        except Exception:
            continue


def _validate_game(game: dict[str, Any]) -> bool:
    try:
        int(game["map"]["id"])
        parse(game["begin_at"])
        team_players: dict[Any, list[Any]] = defaultdict(list)
        for p in game["players"]:
            team_players[p["team"]["id"]].append(p["player"]["id"])
        if len(team_players) != 2:
            return False
        for _, p_ids in team_players.items():
            if len(set(p_ids)) != 5:
                return False
        t1_id, t2_id = list(team_players.keys())
        rounds: list[int] = []
        for r in game["rounds"]:
            if r["round"] is None:
                continue
            if r["winner_team"] not in (t1_id, t2_id):
                return False
            rounds.append(r["round"])
        return min(rounds) == 1 and max(rounds) >= 16
    except Exception:
        return False


def get_game_ids(path_to_dir: str) -> list[int]:
    begin_ats: list[Any] = []
    game_ids: list[int] = []
    for game in generate_game_raw(path_to_dir):
        if _validate_game(game):
            begin_ats.append(parse(game["begin_at"]))
            game_ids.append(int(game["id"]))
    order = np.argsort(begin_ats)
    return np.array(game_ids, dtype=int)[order].tolist()


def get_X_y(
    game_ids: list[int], path_to_dir: str = "data/games_raw"
) -> tuple[np.ndarray, np.ndarray]:
    X: list[list[int]] = []
    y: list[int] = []
    for game_id in game_ids:
        with open(f"{path_to_dir}/{game_id}.json", "r", encoding="utf-8") as f:
            game = json.load(f)
        team_players: dict[Any, list[Any]] = defaultdict(list)
        for p in game["players"]:
            team_players[p["team"]["id"]].append(p["player"]["id"])
        t1_id, t2_id = sorted(team_players.keys())
        X.append(
            [game["map"]["id"], t1_id, t2_id]
            + team_players[t1_id]
            + team_players[t2_id]
        )
        winners = [r["winner_team"] for r in game["rounds"]]
        win_count = Counter(winners)
        y.append(1 if win_count[t1_id] > win_count[t2_id] else 0)
    return np.array(X, dtype=int), np.array(y, dtype=int)
