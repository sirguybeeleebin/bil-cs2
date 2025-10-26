import json

import pytest

from app.ml.data_loader import _generate_game_raw, _validate_game, get_game_ids, get_X_y


@pytest.fixture
def valid_game():
    return {
        "id": 1,
        "map": {"id": 100, "name": "Map1"},
        "begin_at": "2025-01-01T12:00:00Z",
        "players": [
            {"player": {"id": 1}, "team": {"id": 10}},
            {"player": {"id": 2}, "team": {"id": 10}},
            {"player": {"id": 3}, "team": {"id": 10}},
            {"player": {"id": 4}, "team": {"id": 10}},
            {"player": {"id": 5}, "team": {"id": 10}},
            {"player": {"id": 6}, "team": {"id": 20}},
            {"player": {"id": 7}, "team": {"id": 20}},
            {"player": {"id": 8}, "team": {"id": 20}},
            {"player": {"id": 9}, "team": {"id": 20}},
            {"player": {"id": 10}, "team": {"id": 20}},
        ],
        "rounds": [
            {"round": i + 1, "winner_team": 10 if i % 2 == 0 else 20} for i in range(16)
        ],
    }


@pytest.fixture
def invalid_game_missing_team():
    game = {
        "id": 2,
        "map": {"id": 101, "name": "Map2"},
        "begin_at": "2025-01-02T12:00:00Z",
        "players": [
            {"player": {"id": 1}, "team": {"id": 10}},
        ],
        "rounds": [{"round": 1, "winner_team": 10}],
    }
    return game


# -----------------------------
# Tests for generate_game_raw
# -----------------------------


def test_generate_game_raw_reads_json(tmp_path):
    file_path = tmp_path / "game.json"
    data = {"id": 1}
    file_path.write_text(json.dumps(data))

    results = list(_generate_game_raw(str(tmp_path)))
    assert len(results) == 1
    assert results[0]["id"] == 1


def test_generate_game_raw_skips_invalid_json(tmp_path):
    file_path = tmp_path / "bad.json"
    file_path.write_text("{bad json}")

    results = list(_generate_game_raw(str(tmp_path)))
    assert results == []


# -----------------------------
# Tests for _validate_game
# -----------------------------


def test_validate_game_valid(valid_game):
    assert _validate_game(valid_game) is True


def test_validate_game_invalid(invalid_game_missing_team):
    assert _validate_game(invalid_game_missing_team) is False


def test_validate_game_wrong_rounds(valid_game):
    valid_game["rounds"][0]["winner_team"] = 999
    assert _validate_game(valid_game) is False


# -----------------------------
# Tests for get_game_ids
# -----------------------------


def test_get_game_ids(tmp_path, valid_game, invalid_game_missing_team):
    (tmp_path / "1.json").write_text(json.dumps(valid_game))
    (tmp_path / "2.json").write_text(json.dumps(invalid_game_missing_team))

    ids = get_game_ids(str(tmp_path))
    assert ids == [1]  # only valid game


# -----------------------------
# Tests for get_X_y
# -----------------------------


def test_get_X_y(tmp_path, valid_game):
    game_id = valid_game["id"]
    (tmp_path / f"{game_id}.json").write_text(json.dumps(valid_game))

    X, y = get_X_y([game_id], path_to_dir=str(tmp_path))
    assert X.shape == (1, 13)
    assert y.shape == (1,)
    assert y[0] in [0, 1]  # winner encoded


def test_get_X_y_skips_invalid(tmp_path, invalid_game_missing_team):
    game_id = invalid_game_missing_team["id"]
    (tmp_path / f"{game_id}.json").write_text(json.dumps(invalid_game_missing_team))

    X, y = get_X_y([game_id], path_to_dir=str(tmp_path))
    assert X.shape[0] == 0
    assert y.shape[0] == 0
