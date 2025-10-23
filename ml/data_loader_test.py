import json

import numpy as np
import pytest

from ml.data_loader import DataLoader  # adjust import path if needed

# --------------------------
# Sample valid and invalid games
# --------------------------
VALID_GAME = {
    "id": 1,
    "begin_at": "2025-10-23T12:00:00Z",
    "map": {"id": 100},
    "players": [{"team": {"id": 10}, "player": {"id": i}} for i in range(5)]
    + [{"team": {"id": 20}, "player": {"id": i + 5}} for i in range(5)],
    "rounds": [
        {"round": i + 1, "ct": 10, "terrorists": 20, "winner_team": 10}
        for i in range(16)
    ],
}

INVALID_GAME = {
    "id": 2,
    "begin_at": "invalid-date",
    "map": {"id": 100},
    "players": [],
    "rounds": [],
}


# --------------------------
# Fixtures
# --------------------------
@pytest.fixture
def games_dir(tmp_path):
    """Create a temporary directory with some game JSON files."""
    valid_file = tmp_path / "1.json"
    invalid_file = tmp_path / "2.json"
    with open(valid_file, "w") as f:
        json.dump(VALID_GAME, f)
    with open(invalid_file, "w") as f:
        json.dump(INVALID_GAME, f)
    return tmp_path


# --------------------------
# Tests
# --------------------------
def test_get_game_ids_ordered_by_begin_at(games_dir):
    loader = DataLoader(str(games_dir))
    game_ids = loader.get_game_ids_ordered_by_begin_at()
    assert isinstance(game_ids, list)
    assert len(game_ids) == 1  # only valid game
    assert game_ids[0] == VALID_GAME["id"]


def test_get_X_y(games_dir):
    loader = DataLoader(str(games_dir))
    game_ids = loader.get_game_ids_ordered_by_begin_at()
    X, y = loader.get_X_y(game_ids)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == y.shape[0] == 1
    # Feature length: map_id + 2 teams + 10 players
    assert X.shape[1] == 1 + 2 + 10
    # Outcome should be 0 or 1
    assert y[0] in (0, 1)


def test_train_test_split(games_dir):
    loader = DataLoader(str(games_dir))
    game_ids = loader.get_game_ids_ordered_by_begin_at()
    train_ids, test_ids = loader.train_test_split(game_ids, test_size=1)
    # Only 1 game -> train empty, test contains that game
    assert train_ids == []
    assert test_ids == [game_ids[-1]]


def test__validate_game_valid():
    loader = DataLoader("dummy")
    assert loader._validate_game(VALID_GAME) is True


def test__validate_game_invalid():
    loader = DataLoader("dummy")
    assert loader._validate_game(INVALID_GAME) is False


def test__get_game_X_y_shape():
    loader = DataLoader("dummy")
    X, y = loader._get_game_X_y(VALID_GAME)
    assert len(X) == 1 + 2 + 10  # map_id + 2 teams + 10 players
    assert y in (0, 1)
