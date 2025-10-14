import json
import os
from unittest.mock import MagicMock, patch

import pandas as pd
from main import (
    compute_hash,
    fetch_unique_games,
    get_clickhouse_client,
    save_split_json,
    split_train_test,
)

# ------------------------------
# Mock Data
# ------------------------------
sample_df = pd.DataFrame(
    {
        "game_id": list(range(1, 11)),
        "min_begin": pd.date_range("2025-01-01", periods=10),
    }
)

# ------------------------------
# Tests
# ------------------------------


def test_get_clickhouse_client():
    with patch("clickhouse_connect.get_client") as mock_get_client:
        mock_get_client.return_value = "client_instance"
        client = get_clickhouse_client("host", 1234, "user", "pass", "db")
        assert client == "client_instance"
        mock_get_client.assert_called_once_with(
            host="host", port=1234, username="user", password="pass", database="db"
        )


def test_fetch_unique_games():
    mock_client = MagicMock()
    mock_client.query_df.return_value = sample_df
    df = fetch_unique_games(mock_client, "db", "table")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    mock_client.query_df.assert_called_once()
    query = mock_client.query_df.call_args[0][0]
    assert "SELECT DISTINCT game_id" in query


def test_split_train_test_default():
    train_ids, test_ids = split_train_test(sample_df, test_size=3)
    assert train_ids == [1, 2, 3, 4, 5, 6, 7]
    assert test_ids == [8, 9, 10]


def test_split_train_test_empty():
    empty_df = pd.DataFrame(columns=["game_id", "min_begin"])
    train_ids, test_ids = split_train_test(empty_df, test_size=5)
    assert train_ids == []
    assert test_ids == []


def test_compute_hash():
    h = compute_hash(sample_df)
    assert isinstance(h, str)
    assert len(h) == 32  # MD5 hash length


def test_save_split_json(tmp_path):
    train_ids = [1, 2, 3]
    test_ids = [4, 5]
    output_dir = tmp_path
    hash_str = "abc123hash"
    output_path = save_split_json(train_ids, test_ids, output_dir, hash_str)

    # Check file exists
    assert os.path.exists(output_path)

    # Check JSON content
    with open(output_path, "r") as f:
        data = json.load(f)
    assert "train" in data
    assert "test" in data
    assert data["train"] == train_ids
    assert data["test"] == test_ids
    assert "created_at" in data
