from unittest.mock import ANY, MagicMock

import pandas as pd
from main import flatten_game, get_dict, load_to_clickhouse, validate_game_raw

# ---------------------------
# Sample game data
# ---------------------------
sample_game_valid = {
    "id": 1,
    "begin_at": "2025-01-01T12:00:00Z",
    "map": {"id": 5},
    "match": {
        "league": {"id": 10},
        "serie": {"id": 20, "tier": "s"},
        "tournament": {"id": 30},
    },
    "players": [
        {"team": {"id": 100}, "player": {"id": 1001}, "kills": 10},
        {"team": {"id": 100}, "player": {"id": 1002}, "kills": 12},
        {"team": {"id": 100}, "player": {"id": 1003}, "kills": 8},
        {"team": {"id": 100}, "player": {"id": 1004}, "kills": 5},
        {"team": {"id": 100}, "player": {"id": 1005}, "kills": 7},
        {"team": {"id": 200}, "player": {"id": 2001}, "kills": 9},
        {"team": {"id": 200}, "player": {"id": 2002}, "kills": 11},
        {"team": {"id": 200}, "player": {"id": 2003}, "kills": 6},
        {"team": {"id": 200}, "player": {"id": 2004}, "kills": 8},
        {"team": {"id": 200}, "player": {"id": 2005}, "kills": 7},
    ],
    "rounds": [
        {
            "round": i,
            "ct": 100,
            "terrorists": 200,
            "winner_team": 100,
            "outcome": "exploded",
        }
        for i in range(1, 17)
    ],
}

sample_game_invalid = {"id": 2, "begin_at": "invalid-date", "players": [], "rounds": []}


# ---------------------------
# Tests
# ---------------------------
def test_validate_game_raw_valid():
    assert validate_game_raw(sample_game_valid) is True


def test_validate_game_raw_invalid():
    assert validate_game_raw(sample_game_invalid) is False


def test_flatten_game_structure():
    flat = flatten_game(sample_game_valid)
    assert isinstance(flat, list)
    # 2 teams × 5 players × 5 opponents × 16 rounds
    assert len(flat) == 2 * 5 * 5 * 16
    for row in flat:
        assert "game_id" in row
        assert "player_id" in row
        assert "round_id" in row
        assert "round_win" in row


def test_get_dict():
    d = {"a": {"b": 1}}
    assert get_dict(d, "a") == {"b": 1}
    assert get_dict(d, "missing") == {}


def test_load_to_clickhouse_calls(monkeypatch):
    # Mock client
    mock_client = MagicMock()

    # Generate sample flattened data
    flat_data = flatten_game(sample_game_valid)

    # Patch pd.DataFrame.from_records to monitor insertion
    monkeypatch.setattr(
        "pandas.DataFrame.from_records", lambda x: pd.DataFrame(flat_data)
    )

    load_to_clickhouse(
        mock_client, flat_data, table_name="test_table", database="test_db"
    )

    # Ensure table creation command called
    mock_client.command.assert_called()
    # Ensure insert called
    mock_client.insert.assert_called_with("test_db.test_table", ANY, column_names=ANY)
