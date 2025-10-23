import json
from unittest.mock import MagicMock

import pytest

from etl.etl import (
    _extract_map,
    _extract_players,
    _extract_teams,
    _generate_game_raw,
    _send_data,
    load_cs2_data,
)

# ----------------------------
# Fixtures
# ----------------------------


@pytest.fixture
def sample_game():
    return {
        "map": {"id": 1, "name": "Dust2"},
        "players": [
            {
                "player": {"id": 101, "name": "Alice"},
                "team": {"id": 201, "name": "TeamA"},
            },
            {
                "player": {"id": 102, "name": "Bob"},
                "team": {"id": 201, "name": "TeamA"},
            },
        ],
    }


@pytest.fixture
def temp_game_file(tmp_path, sample_game):
    file_path = tmp_path / "game1.json"
    file_path.write_text(json.dumps(sample_game))
    return tmp_path


@pytest.fixture
def temp_bad_file(tmp_path):
    file_path = tmp_path / "bad.json"
    file_path.write_text("{invalid_json")
    return tmp_path


# ----------------------------
# Test _generate_game_raw
# ----------------------------


def test_generate_game_raw_yields_file(temp_game_file, sample_game):
    results = list(_generate_game_raw(str(temp_game_file)))
    assert len(results) == 1
    file_path, game = results[0]
    assert game == sample_game
    assert file_path.endswith("game1.json")


def test_generate_game_raw_skips_invalid_file(temp_bad_file):
    results = list(_generate_game_raw(str(temp_bad_file)))
    assert results == []


# ----------------------------
# Test _extract_map
# ----------------------------


def test_extract_map_returns_correct_map(sample_game):
    map_data = _extract_map(sample_game)
    assert map_data == {"map_id": 1, "name": "Dust2"}


def test_extract_map_handles_missing_map():
    map_data = _extract_map({"players": []})
    assert map_data is None


# ----------------------------
# Test _extract_teams
# ----------------------------


def test_extract_teams_returns_unique_teams(sample_game):
    teams = _extract_teams(sample_game)
    assert teams == [{"team_id": 201, "name": "TeamA"}]


def test_extract_teams_handles_no_teams():
    teams = _extract_teams({"players": []})
    assert teams == []


# ----------------------------
# Test _extract_players
# ----------------------------


def test_extract_players_returns_unique_players(sample_game):
    players = _extract_players(sample_game)
    expected = [
        {"player_id": 101, "name": "Alice"},
        {"player_id": 102, "name": "Bob"},
    ]
    assert players == expected


def test_extract_players_handles_no_players():
    players = _extract_players({"players": []})
    assert players == []


# ----------------------------
# Test _send_data
# ----------------------------


def test_send_data_calls_post():
    mock_client = MagicMock()
    data = [{"id": 1}]
    _send_data(mock_client, "maps", data, "http://testserver")
    mock_client.post.assert_called_once_with("http://testserver/maps/save", json=data)


def test_send_data_skips_empty():
    mock_client = MagicMock()
    _send_data(mock_client, "maps", [], "http://testserver")
    mock_client.post.assert_not_called()


# ----------------------------
# Test load_cs2_data
# ----------------------------


def test_load_cs2_data_calls_send_data(temp_game_file):
    mock_client = MagicMock()
    result = load_cs2_data(str(temp_game_file), "http://testserver", client=mock_client)
    assert result["total"] == 1
    assert result["success"] == 1
    assert result["error"] == 0
    # Should call for maps, teams, players
    assert mock_client.post.call_count == 3


def test_load_cs2_data_handles_invalid_file(temp_bad_file):
    mock_client = MagicMock()
    result = load_cs2_data(str(temp_bad_file), "http://testserver", client=mock_client)
    assert result["total"] == 0
    assert result["success"] == 0
    assert result["error"] == 0
    mock_client.post.assert_not_called()
