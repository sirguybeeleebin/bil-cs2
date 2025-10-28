import json

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
    assert _validate_game(sample_game) is True

    # Remove one player -> invalid
    sample_game_invalid = sample_game.copy()
    sample_game_invalid["players"] = sample_game_invalid["players"][:-1]
    assert _validate_game(sample_game_invalid) is False


def test_get_game_ids(tmp_path, sample_game, caplog):
    file_path = tmp_path / "game1.json"
    file_path.write_text(json.dumps(sample_game))

    with caplog.at_level("INFO"):
        ids = get_game_ids(str(tmp_path))
    assert ids == [1]
    assert "Загружено 1 корректных игр" in caplog.text


def test_get_X_y(tmp_path, sample_game):
    file_path = tmp_path / "1.json"
    file_path.write_text(json.dumps(sample_game))

    X, y = get_X_y([1], path_to_dir=str(tmp_path))
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == 1
    assert y.shape[0] == 1
    # The first team wins 8 rounds, second team wins 8 rounds -> tie -> y=0
    assert y[0] == 0
