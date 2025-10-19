import json
import os
import warnings
from collections import defaultdict

import numpy as np
from dateutil.parser import parse


warnings.filterwarnings("ignore")


def generate_game_raw(path_to_games_raw_dir="data/games_raw"):
    for filename in os.listdir(path_to_games_raw_dir):
        file_path = os.path.join(path_to_games_raw_dir, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                yield json.load(f)
        except:
            continue


def validate_game(game):
    try:
        int(game["id"])
        parse(game["begin_at"])
        int(game["match"]["league"]["id"])
        int(game["match"]["serie"]["id"])
        int(game["match"]["tournament"]["id"])
        int(game["map"]["id"])
        team_players = defaultdict(list)
        for p in game["players"]:
            team_players[p["team"]["id"]].append(p["player"]["id"])
        if len(team_players) != 2:
            return False
        for p_ids in team_players.values():
            if len(set(p_ids)) != 5:
                return False
        team_ids = list(team_players.keys())
        rounds = []
        for r in game["rounds"]:
            if (
                r["round"] is None
                or r["ct"] not in team_ids
                or r["terrorists"] not in team_ids
                or r["winner_team"] not in team_ids
            ):
                return False
            rounds.append(r["round"])
        if min(rounds) != 1 or max(rounds) < 16:
            return False
        return True
    except:
        return False


def get_game_ids(path_to_games_raw_dir="data/games_raw"):
    game_ids_valid, game_begin_at_valid = [], []
    for game in generate_game_raw(path_to_games_raw_dir):
        if validate_game(game):
            game_ids_valid.append(game["id"])
            game_begin_at_valid.append(parse(game["begin_at"]))
    return np.array(game_ids_valid)[np.argsort(game_begin_at_valid)].tolist()


def get_X_y(path_to_games_raw, game_ids):
    X, y = [], []
    for game_id in game_ids:
        file_path = os.path.join(path_to_games_raw, f"{game_id}.json")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                game = json.load(f)
            team_players = defaultdict(list)            
            for p in game["players"]:
                team_players[p["team"]["id"]].append(p["player"]["id"])                
            t1_id, t2_id = sorted(team_players.keys())
            p_ids = sorted(team_players[t1_id]) + sorted(team_players[t2_id])
            X.append(
                [
                    parse(game["begin_at"]),
                    game["map"]["id"],
                    int(game["rounds"][0]["ct"] == t1_id),
                    t1_id,
                    t2_id,
                ]
                + p_ids                
            )
            team_win_count = {t1_id: 0, t2_id: 0}
            for r in game["rounds"]:
                team_win_count[r["winner_team"]] += 1
            y.append(int(team_win_count[t1_id] > team_win_count[t2_id]))
        except:
            continue
    return np.array(X), np.array(y)
