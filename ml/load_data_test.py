import json
import os
import tempfile

import pytest
from dateutil.parser import parse

from ml.load_data import generate_game_raw, get_game_ids, get_X_y, validate_game


# Helper to create a valid game structure
def create_valid_game(game_id=1):
    game = {
        "id": str(game_id),
        "begin_at": "2025-01-01T12:00:00Z",
        "map": {"id": 1},
        "match": {
            "league": {"id": 10},
            "serie": {"id": 20, "tier": "Pro"},
            "tournament": {"id": 30},
        },
        "players": [],
        "rounds": [],
    }
    t1_id, t2_id = 100, 200
    for i in range(5):
        game["players"].append({"team": {"id": t1_id}, "player": {"id": i + 1}})
        game["players"].append({"team": {"id": t2_id}, "player": {"id": i + 6}})
    # Add 16 rounds
    for i in range(16):
        round_info = {
            "round": i + 1,
            "ct": t1_id if i % 2 == 0 else t2_id,
            "terrorists": t2_id if i % 2 == 0 else t1_id,
            "winner_team": t1_id if i < 9 else t2_id,  # 9 rounds team1 wins
        }
        game["rounds"].append(round_info)
    return game


@pytest.fixture
def temp_games_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_generate_game_raw_yields_dicts(temp_games_dir):
    game = create_valid_game()
    file_path = os.path.join(temp_games_dir, "1.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(game, f)

    result = list(generate_game_raw(temp_games_dir))
    assert len(result) == 1
    assert isinstance(result[0], dict)
    assert result[0]["id"] == "1"


def test_validate_game_valid_and_invalid():
    game = create_valid_game()
    assert validate_game(game) is True

    # Invalid: missing players
    invalid_game = create_valid_game()
    invalid_game["players"] = []
    assert validate_game(invalid_game) is False

    # Invalid: less than 16 rounds
    invalid_game2 = create_valid_game()
    invalid_game2["rounds"] = invalid_game2["rounds"][:10]
    assert validate_game(invalid_game2) is False


def test_get_game_ids_returns_sorted(temp_games_dir):
    game1 = create_valid_game(2)
    game2 = create_valid_game(1)
    game1["begin_at"] = "2025-01-02T12:00:00Z"
    game2["begin_at"] = "2025-01-01T12:00:00Z"

    for g in [game1, game2]:
        file_path = os.path.join(temp_games_dir, f"{g['id']}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(g, f)

    ids = get_game_ids(temp_games_dir)
    assert ids == ["1", "2"]


def test_get_X_y_shapes_and_values(temp_games_dir):
    game = create_valid_game()
    file_path = os.path.join(temp_games_dir, f"{game['id']}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(game, f)

    X, y = get_X_y(temp_games_dir, [game["id"]])
    assert X.shape[0] == 1
    assert y.shape[0] == 1

    # Check map ID and match info
    assert X[0][5] == 1  # map id
    assert X[0][1] == "10"  # league id
    assert X[0][2] == "20"  # serie id
    assert X[0][3] == "30"  # tournament id
    assert X[0][4] == "Pro"  # tier

    # Check team IDs
    assert X[0][7] == 100
    assert X[0][8] == 200

    # Check player IDs
    assert set(X[0][9:14]) == set(range(1, 6))  # team1
    assert set(X[0][14:19]) == set(range(6, 11))  # team2

    # Check y: team1 wins 9 rounds, team2 wins 7 rounds -> team1 wins
    assert y[0] == 1
