import argparse
import hashlib
import json
import logging
import os
from datetime import datetime

import clickhouse_connect

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("cs2_split")


# -------------------------------
# Argument Parsing
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/Test split for CS2 games (no validation)"
    )
    parser.add_argument("--clickhouse_host", type=str, default="localhost")
    parser.add_argument("--clickhouse_port", type=int, default=8123)
    parser.add_argument("--clickhouse_user", type=str, default="cs2_user")
    parser.add_argument("--clickhouse_password", type=str, default="cs2_password")
    parser.add_argument("--clickhouse_db", type=str, default="cs2_db")
    parser.add_argument("--table_name", type=str, default="games_flat")
    parser.add_argument(
        "--test_size", type=int, default=100, help="Number of last games for test split"
    )
    parser.add_argument("--output_dir", type=str, default="data/train_test_splits")
    return parser.parse_args()


# -------------------------------
# ClickHouse Functions
# -------------------------------
def get_clickhouse_client(host, port, user, password, database):
    client = clickhouse_connect.get_client(
        host=host, port=port, username=user, password=password, database=database
    )
    log.info(f"Connected to ClickHouse at {host}:{port}")
    return client


def fetch_unique_games(client, database, table_name):
    query = f"""
        SELECT DISTINCT game_id, MIN(begin_at) as min_begin
        FROM {database}.{table_name}
        GROUP BY game_id
        ORDER BY min_begin ASC
    """
    df_games = client.query_df(query)
    log.info(f"Fetched {len(df_games)} unique games")
    return df_games


# -------------------------------
# Split Functions
# -------------------------------
def split_train_test(df_games, test_size):
    total_games = len(df_games)
    if total_games == 0:
        log.warning("No games found. Returning empty splits.")
        return [], []

    test_df = df_games.tail(test_size)
    train_df = df_games.head(total_games - test_size)
    log.info(f"Split complete: Train={len(train_df)}, Test={len(test_df)}")
    return train_df["game_id"].astype(int).tolist(), test_df["game_id"].astype(
        int
    ).tolist()


def compute_hash(df_games):
    all_ids_str = ",".join(df_games["game_id"].astype(str))
    return hashlib.md5(all_ids_str.encode("utf-8")).hexdigest()


# -------------------------------
# Output Functions
# -------------------------------
def save_split_json(train_ids, test_ids, output_dir, hash_str):
    os.makedirs(output_dir, exist_ok=True)
    split_dict = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train": train_ids,
        "test": test_ids,
    }
    output_path = os.path.join(output_dir, f"{hash_str}.json")
    with open(output_path, "w") as f:
        json.dump(split_dict, f, indent=4)
    log.info(f"Split saved to {output_path}")
    return output_path


# -------------------------------
# Main
# -------------------------------
def main():
    args = parse_args()
    client = get_clickhouse_client(
        host=args.clickhouse_host,
        port=args.clickhouse_port,
        user=args.clickhouse_user,
        password=args.clickhouse_password,
        database=args.clickhouse_db,
    )

    df_games = fetch_unique_games(client, args.clickhouse_db, args.table_name)
    train_ids, test_ids = split_train_test(df_games, args.test_size)

    if not train_ids and not test_ids:
        log.info("No games to save. Exiting.")
        return

    hash_str = compute_hash(df_games)
    save_split_json(train_ids, test_ids, args.output_dir, hash_str)
    log.info(f"✅ Done at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
