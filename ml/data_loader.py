import json
import logging
import os
import warnings
from collections import Counter, defaultdict

import numpy as np
from dateutil.parser import parse

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def generate_game_raw(path_to_games_raw_dir: str):
    for root, _, files in os.walk(path_to_games_raw_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                yield data
            except Exception:
                continue


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
    except:
        return False


def get_game_ids_ordered_by_begin_at(path_to_games_raw_dir: str):
    total = 0
    valid = 0
    invalid = 0
    begin_ats = []
    game_ids = []
    for game in generate_game_raw(path_to_games_raw_dir):
        total += 1
        if validate_game(game):
            valid += 1
            begin_ats.append(parse(game["begin_at"]))
            game_ids.append(int(game["id"]))
        else:
            invalid += 1
        log.info(f"Обработано {total} игр (валидных: {valid}, невалидных: {invalid})")
    ordered_ids = np.array(game_ids)[np.argsort(begin_ats)].tolist()
    return ordered_ids


def get_game_X_y(game: dict):
    map_id = int(game["map"]["id"])
    team_players = defaultdict(list)
    for p in game["players"]:
        team_players[p["team"]["id"]].append(p["player"]["id"])
    t1_id, t2_id = sorted(team_players.keys())
    win_counts = Counter()
    for r in game["rounds"]:
        win_counts[r["winner_team"]] += 1
    y = int(win_counts[t1_id] > win_counts[t2_id])
    X = (
        [map_id]
        + [t1_id, t2_id]
        + sorted(team_players[t1_id])
        + sorted(team_players[t2_id])
    )
    return X, y


def get_X_y(path_to_games_raw_dir: str, game_ids: list[int]):
    X, y = [], []
    for game_id in game_ids:
        with open(os.path.join(path_to_games_raw_dir, f"{game_id}.json"), "r") as f:
            game = json.load(f)
        _x, _y = get_game_X_y(game)
        X.append(_x)
        y.append(_y)
    return np.array(X), np.array(y)
