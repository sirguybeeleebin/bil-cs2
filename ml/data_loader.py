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


class DataLoader:
    def __init__(self, path_to_games_raw_dir: str):
        self.path_to_games_raw_dir = path_to_games_raw_dir

    def get_game_ids_ordered_by_begin_at(self):
        begin_ats, game_ids = [], []
        for game in self._generate_game_raw():
            if self._validate_game(game):
                begin_ats.append(parse(game["begin_at"]))
                game_ids.append(int(game["id"]))
        return np.array(game_ids)[np.argsort(begin_ats)].tolist()

    def _generate_game_raw(self):
        for root, _, files in os.walk(self.path_to_games_raw_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    yield data
                except (json.JSONDecodeError, OSError) as e:
                    log.warning("Failed to load file %s: %s", file_path, e)
                    continue

    def _validate_game(self, game: dict) -> bool:
        try:
            int(game["map"]["id"])
            parse(game["begin_at"])
            team_players = defaultdict(list)
            for p in game["players"]:
                team_players[p["team"]["id"]].append(p["player"]["id"])
            if len(team_players) != 2:
                return False
            for _, p_ids in team_players.items():
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
            if min(rounds) != 1 or max(rounds) < 16:
                return False
            return True
        except (KeyError, ValueError, TypeError) as e:
            log.warning("Invalid game data: %s", e)
            return False

    def get_X_y(self, game_ids: list[int]):
        X, y = [], []
        for game_id in game_ids:
            file_path = os.path.join(self.path_to_games_raw_dir, f"{game_id}.json")
            with open(file_path, "r") as f:
                game = json.load(f)
            _x, _y = self._get_game_X_y(game)
            X.append(_x)
            y.append(_y)
        return np.array(X), np.array(y)

    def _get_game_X_y(self, game: dict):
        map_id = int(game["map"]["id"])
        team_players = defaultdict(list)
        for p in game["players"]:
            team_players[p["team"]["id"]].append(p["player"]["id"])
        t1_id, t2_id = sorted(team_players.keys())
        win_counts = Counter(r["winner_team"] for r in game["rounds"])
        return (
            [map_id]
            + [t1_id, t2_id]
            + sorted(team_players[t1_id])
            + sorted(team_players[t2_id]),
            int(win_counts[t1_id] > win_counts[t2_id]),
        )

    def train_test_split(self, game_ids: list[int], test_size: int = 100):
        if len(game_ids) < test_size:
            raise ValueError("Not enough games to create a test set of this size.")
        return game_ids[:-test_size], game_ids[-test_size:]
