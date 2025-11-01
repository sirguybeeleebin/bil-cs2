import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
from dateutil.parser import parse


def _validate_game(game: dict[str, Any]) -> bool:
    try:
        int(game["map"]["id"])
        parse(game["begin_at"])
        team_players: dict[int, list[int]] = defaultdict(list)
        for p in game["players"]:
            team_players[p["team"]["id"]].append(p["player"]["id"])
        if len(team_players) != 2:
            return False
        for p_ids in team_players.values():
            if len(set(p_ids)) != 5:
                return False
        t1_id, t2_id = list(team_players.keys())
        rounds: list[int] = []
        for r in game.get("rounds", []):
            if r.get("round") is None:
                continue
            if r.get("winner_team") not in (t1_id, t2_id):
                return False
            if r.get("ct") not in (t1_id, t2_id):
                return False
            if r.get("terrorists") not in (t1_id, t2_id):
                return False
            rounds.append(r["round"])
        if not rounds or min(rounds) != 1 or max(rounds) < 16:
            return False
        return True
    except Exception:
        return False


def get_game_ids(data_dir: Path) -> List[str]:
    begin_ats: List[datetime] = []
    game_ids: List[str] = []

    for fpath in data_dir.glob("*.json"):
        try:
            with fpath.open("r", encoding="utf-8") as f:
                game = json.load(f)
        except Exception:
            continue

        if _validate_game(game):
            begin_ats.append(
                datetime.fromisoformat(game["begin_at"].replace("Z", "+00:00"))
            )
            game_ids.append(str(game["id"]))

    if not game_ids:
        return []

    sorted_indices = np.argsort(begin_ats)
    return [game_ids[i] for i in sorted_indices], min(begin_ats), max(begin_ats)


def build_dataset(
    path_to_dir: Path, game_ids: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    X: List[List[int]] = []
    y: List[int] = []

    for game_id in game_ids:
        fpath = path_to_dir / f"{game_id}.json"
        if not fpath.exists():
            continue
        try:
            with fpath.open("r", encoding="utf-8") as f:
                game = json.load(f)

            team_players: defaultdict[int, list[int]] = defaultdict(list)
            for p in game["players"]:
                team_players[p["team"]["id"]].append(p["player"]["id"])

            t1_id, t2_id = sorted(team_players.keys())
            X.append(
                [game["map"]["id"], t1_id, t2_id]
                + team_players[t1_id]
                + team_players[t2_id]
            )

            winners = [r["winner_team"] for r in game.get("rounds", [])]
            win_count = Counter(winners)
            y.append(1 if win_count[t1_id] > win_count[t2_id] else 0)
        except Exception:
            continue

    return np.array(X), np.array(y)
