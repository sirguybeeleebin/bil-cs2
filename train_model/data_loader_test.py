import json
from copy import deepcopy

import numpy as np
import pytest

from train_model.data_loader import (
    _generate_game_raw,
    _validate_game,
    get_game_ids,
    get_X_y,
)


@pytest.fixture
def sample_game():
    return {
        "id": 1,
        "map": {"id": 100, "name": "Dust II"},
        "begin_at": "2025-10-28T12:00:00Z",
        "players": [
            {
                "player": {"id": 101, "name": "Alice"},
                "team": {"id": 10, "name": "Team A"},
            },
            {
                "player": {"id": 102, "name": "Bob"},
                "team": {"id": 10, "name": "Team A"},
            },
            {
                "player": {"id": 103, "name": "Eve"},
                "team": {"id": 10, "name": "Team A"},
            },
            {
                "player": {"id": 104, "name": "Mallory"},
                "team": {"id": 10, "name": "Team A"},
            },
            {
                "player": {"id": 105, "name": "Trent"},
                "team": {"id": 10, "name": "Team A"},
            },
            {
                "player": {"id": 201, "name": "Charlie"},
                "team": {"id": 20, "name": "Team B"},
            },
            {
                "player": {"id": 202, "name": "David"},
                "team": {"id": 20, "name": "Team B"},
            },
            {
                "player": {"id": 203, "name": "Frank"},
                "team": {"id": 20, "name": "Team B"},
            },
            {
                "player": {"id": 204, "name": "Grace"},
                "team": {"id": 20, "name": "Team B"},
            },
            {
                "player": {"id": 205, "name": "Heidi"},
                "team": {"id": 20, "name": "Team B"},
            },
        ],
        "rounds": [
            {"round": i, "winner_team": 10 if i % 2 == 0 else 20} for i in range(1, 17)
        ],
    }


def test_generate_game_raw(tmp_path, sample_game):
    file_path = tmp_path / "game1.json"
    file_path.write_text(json.dumps(sample_game))

    games = list(_generate_game_raw(str(tmp_path)))
    assert len(games) == 1
    assert games[0]["id"] == 1


def test_validate_game(sample_game):
    # Valid game
    assert _validate_game(sample_game) is True

    # Remove one player -> invalid
    invalid_game = deepcopy(sample_game)
    invalid_game["players"] = invalid_game["players"][:-1]
    assert _validate_game(invalid_game) is False

    # Invalid number of teams
    invalid_game2 = deepcopy(sample_game)
    invalid_game2["players"] = invalid_game2["players"][:5]  # only one team
    assert _validate_game(invalid_game2) is False

    # Invalid round winner
    invalid_game3 = deepcopy(sample_game)
    invalid_game3["rounds"][0]["winner_team"] = 999
    assert _validate_game(invalid_game3) is False


def test_get_game_ids(tmp_path, sample_game):
    file_path = tmp_path / "game1.json"
    file_path.write_text(json.dumps(sample_game))

    # Just check the output, not the logs
    ids = get_game_ids(str(tmp_path))
    assert ids == [1]


def test_get_X_y(tmp_path, sample_game):
    file_path = tmp_path / "1.json"
    file_path.write_text(json.dumps(sample_game))

    X, y = get_X_y([1], path_to_dir=str(tmp_path))
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == 1
    assert y.shape[0] == 1

    # Team 10 wins 8 rounds, Team 20 wins 8 rounds -> tie -> y=0
    assert y[0] == 0
    # Check feature vector length: map + 2 team ids + 10 player ids
    assert X.shape[1] == 1 + 2 + 10
