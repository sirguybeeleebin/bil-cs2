import os
import json
from collections import defaultdict
from dateutil.parser import parse
import pandas as pd
import numpy as np
import hashlib
import argparse

def hash_ids(game_ids_list: list) -> str:
    """Return SHA256 hash of concatenated game IDs."""
    # Convert all IDs to strings
    concat_ids = ",".join(map(str, game_ids_list))
    return hashlib.sha256(concat_ids.encode("utf-8")).hexdigest()[:16]

def train_test_split_games(path_to_raw_dir: str, path_to_split_dir: str, test_size: int):
    begins_at = []
    game_ids = []

    os.makedirs(path_to_split_dir, exist_ok=True)

    for game_file in os.listdir(path_to_raw_dir):
        try:
            with open(os.path.join(path_to_raw_dir, game_file), "r", encoding="utf-8") as f:
                game_data = json.load(f)

            game_begin = parse(game_data["begin_at"])
            dd = defaultdict(list)
            for p in game_data["players"]:
                dd[p["team"]["id"]].append(p["player"]["id"])

            if len(dd) != 2 or any(len(players) != 5 for players in dd.values()):
                continue

            t1_id, t2_id = sorted(dd.keys())
            rounds_df = pd.DataFrame.from_records(game_data["rounds"])
            if "winner_team" not in rounds_df:
                continue

            winner_team = rounds_df["winner_team"].value_counts().idxmax()
            if winner_team not in (t1_id, t2_id):
                continue

            game_ids.append(game_data["id"])
            begins_at.append(game_begin)

        except Exception:
            continue

    if not game_ids:
        return
        
    order = np.argsort(begins_at)
    game_ids_sorted = np.array(game_ids)[order]

    # Split train/test
    games_train = game_ids_sorted[:-test_size].tolist()
    games_test = game_ids_sorted[-test_size:].tolist()

    # Generate hash for filename using all game_ids
    hash_id = hash_ids(game_ids_sorted.tolist())

    # Save to JSON file named by the hash
    output_file = os.path.join(path_to_split_dir, f"{hash_id}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"train": games_train, "test": games_test}, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split game JSONs into train/test sets and save with hashed filename."
    )
    parser.add_argument(
        "--path_to_games_raw_dir",
        type=str,
        default="data/games_raw",
        help="Path to directory containing raw game JSON files."
    )
    parser.add_argument(
        "--path_to_train_test_splits_dir",
        type=str,
        default="data/train_test_splits",
        help="Directory to save train/test split JSON."
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=100,
        help="Number of games to use for the test set (integer)."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_test_split_games(
        args.path_to_games_raw_dir,
        args.path_to_train_test_splits_dir,
        args.test_size
    )


if __name__ == "__main__":
    main()
