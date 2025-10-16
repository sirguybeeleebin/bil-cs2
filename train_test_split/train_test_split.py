import json
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from clickhouse_driver import Client

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def fetch_game_ids(client: Client) -> list[int]:
    query = """
    SELECT DISTINCT game_id
    FROM games_flatten
    ORDER BY begin_at ASC
    """
    return [row[0] for row in client.execute(query)]


def create_split(game_ids: list[int], output_dir: Path):
    test_game_ids = game_ids[-100:]
    train_game_ids = game_ids[:-100]

    hash_input = ",".join(str(gid) for gid in game_ids)
    hash_id = hashlib.md5(hash_input.encode("utf-8")).hexdigest()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{hash_id}.json"

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train": train_game_ids,
        "test": test_game_ids
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return str(out_file), hash_id, len(train_game_ids), len(test_game_ids)


def train_test_split(
    clickhouse_host: str = "localhost",
    clickhouse_port: int = 9000,
    clickhouse_user: str = "cs2_user",
    clickhouse_password: str = "cs2_password",
    clickhouse_db: str = "cs2_db",
    output_dir: str = "data/train_test_splits",
):
    """Run train/test split with explicit arguments."""
    client = Client(
        host=clickhouse_host,
        port=clickhouse_port,
        user=clickhouse_user,
        password=clickhouse_password,
        database=clickhouse_db,
    )
    log.info("Connected to ClickHouse")

    game_ids = fetch_game_ids(client)
    log.info(f"Fetched {len(game_ids)} unique game_ids")

    out_file, hash_id, train_len, test_len = create_split(game_ids, Path(output_dir))
    log.info(f"Train/test split saved to {out_file} (train={train_len}, test={test_len})")
    log.info("Process finished")
    
if __name__ == "__main__":
    train_test_split()
    
