import json

import pytest

from ml.dataset_loader import _validate_game, build_dataset, get_game_ids


# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def valid_game_dict():
    return {
        "id": 1,
        "map": {"id": 101},
        "begin_at": "2025-11-04T12:00:00Z",
        "players": [
            {"team": {"id": 10}, "player": {"id": 1}},
            {"team": {"id": 10}, "player": {"id": 2}},
            {"team": {"id": 10}, "player": {"id": 3}},
            {"team": {"id": 10}, "player": {"id": 4}},
            {"team": {"id": 10}, "player": {"id": 5}},
            {"team": {"id": 20}, "player": {"id": 6}},
            {"team": {"id": 20}, "player": {"id": 7}},
            {"team": {"id": 20}, "player": {"id": 8}},
            {"team": {"id": 20}, "player": {"id": 9}},
            {"team": {"id": 20}, "player": {"id": 10}},
        ],
        "rounds": [
            {"round": 1, "winner_team": 10, "ct": 10, "terrorists": 20},
            {"round": 2, "winner_team": 20, "ct": 20, "terrorists": 10},
            {"round": 16, "winner_team": 10, "ct": 10, "terrorists": 20},
        ],
    }


# -----------------------------
# _validate_game tests
# -----------------------------
def test_validate_game_valid(valid_game_dict):
    assert _validate_game(valid_game_dict) is True


def test_validate_game_invalid_team_count(valid_game_dict):
    game = valid_game_dict.copy()
    game["players"] = game["players"][:-1]  # remove a player to make teams <5
    assert _validate_game(game) is False


def test_validate_game_invalid_rounds(valid_game_dict):
    game = valid_game_dict.copy()
    game["rounds"][0]["winner_team"] = 999  # invalid team
    assert _validate_game(game) is False


def test_validate_game_empty_rounds(valid_game_dict):
    game = valid_game_dict.copy()
    game["rounds"] = []
    assert _validate_game(game) is False


def test_validate_game_invalid_ct_terrorists(valid_game_dict):
    game = valid_game_dict.copy()
    game["rounds"][0]["ct"] = 999
    game["rounds"][0]["terrorists"] = 999
    assert _validate_game(game) is False


# -----------------------------
# get_game_ids tests
# -----------------------------
def test_get_game_ids(tmp_path, valid_game_dict):
    # Create multiple JSON files
    game1 = valid_game_dict.copy()
    game1["id"] = 1
    game1["begin_at"] = "2025-11-04T10:00:00Z"

    game2 = valid_game_dict.copy()
    game2["id"] = 2
    game2["begin_at"] = "2025-11-04T12:00:00Z"

    for g in [game1, game2]:
        fpath = tmp_path / f"{g['id']}.json"
        fpath.write_text(json.dumps(g))

    ids, min_time, max_time = get_game_ids(tmp_path)
    assert ids == ["1", "2"]  # should be sorted by begin_at
    assert min_time.isoformat() == "2025-11-04T10:00:00+00:00"
    assert max_time.isoformat() == "2025-11-04T12:00:00+00:00"


def test_get_game_ids_no_valid(tmp_path):
    # No valid JSON files
    fpath = tmp_path / "bad.json"
    fpath.write_text("not a json")
    ids = get_game_ids(tmp_path)
    assert ids == []


# -----------------------------
# build_dataset tests
# -----------------------------
def test_build_dataset(tmp_path, valid_game_dict):
    game = valid_game_dict.copy()
    game["id"] = 1
    fpath = tmp_path / "1.json"
    fpath.write_text(json.dumps(game))

    X, y = build_dataset(tmp_path, [1])
    assert X.shape == (1, 13)  # 1 map + 2 team IDs + 10 players
    assert y.shape == (1,)

    # Проверяем значения
    assert X[0][0] == 101  # map id
    assert set(X[0][1:3]) == {10, 20}  # team ids
    assert set(X[0][3:8]) == {1, 2, 3, 4, 5}  # players team1
    assert set(X[0][8:13]) == {6, 7, 8, 9, 10}  # players team2
    assert y[0] in (0, 1)


def test_build_dataset_missing_file(tmp_path):
    X, y = build_dataset(tmp_path, [999])  # file does not exist
    assert X.shape == (0,)
    assert y.shape == (0,)
