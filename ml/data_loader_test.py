import json

import pytest

from ml.data_loader import _validate_game, generate_game_raw, get_game_ids, get_X_y


@pytest.fixture
def sample_game_valid():
    # 16 rounds to satisfy _validate_game
    rounds = [
        {"round": i + 1, "winner_team": 10 if i % 2 == 0 else 20} for i in range(16)
    ]
    return {
        "id": 123,
        "map": {"id": 1},
        "begin_at": "2025-10-25T12:00:00Z",
        "players": [{"team": {"id": 10}, "player": {"id": i}} for i in range(5)]
        + [{"team": {"id": 20}, "player": {"id": i}} for i in range(5, 10)],
        "rounds": rounds,
    }


@pytest.fixture
def sample_game_invalid():
    # Invalid date, empty players, no rounds
    return {
        "id": 124,
        "map": {"id": 1},
        "begin_at": "invalid-date",
        "players": [],
        "rounds": [],
    }


def test_generate_game_raw(tmp_path, sample_game_valid):
    file_path = tmp_path / "1.json"
    file_path.write_text(json.dumps(sample_game_valid))
    results = list(generate_game_raw(str(tmp_path)))
    assert len(results) == 1
    assert results[0]["id"] == 123


def test_validate_game(sample_game_valid, sample_game_invalid):
    assert _validate_game(sample_game_valid) is True
    assert _validate_game(sample_game_invalid) is False


def test_get_game_ids(tmp_path, sample_game_valid):
    file_path = tmp_path / "123.json"
    file_path.write_text(json.dumps(sample_game_valid))
    game_ids = get_game_ids(str(tmp_path))
    assert isinstance(game_ids, list)
    assert 123 in game_ids


def test_get_X_y(tmp_path, sample_game_valid):
    file_path = tmp_path / "123.json"
    file_path.write_text(json.dumps(sample_game_valid))
    X, y = get_X_y([123], path_to_dir=str(tmp_path))
    assert X.shape == (1, 13)  # 1 map + 2 teams + 10 players
    assert y.shape == (1,)
    assert y[0] in (0, 1)


def test_validate_game_missing_player(tmp_path):
    game = {
        "id": 125,
        "map": {"id": 1},
        "begin_at": "2025-10-25T12:00:00Z",
        "players": [{"team": {"id": 10}, "player": {"id": 0}}],  # not enough players
        "rounds": [{"round": i + 1, "winner_team": 10} for i in range(16)],
    }
    assert _validate_game(game) is False


def test_validate_game_round_winner_invalid(tmp_path, sample_game_valid):
    game = sample_game_valid.copy()
    game["rounds"][0]["winner_team"] = 999  # invalid winner_team
    assert _validate_game(game) is False
