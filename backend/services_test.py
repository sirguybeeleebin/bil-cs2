import json
from unittest.mock import MagicMock

import pytest

from backend.repositories import MapRepository, PlayerRepository, TeamRepository
from backend.services import DictionaryService, ForecasterService


# ------------------------------
# Fixtures
# ------------------------------
@pytest.fixture
def mock_repositories():
    return {
        "map_repository": MagicMock(spec=MapRepository),
        "team_repository": MagicMock(spec=TeamRepository),
        "player_repository": MagicMock(spec=PlayerRepository),
    }


@pytest.fixture
def dictionary_service(mock_repositories):
    return DictionaryService(
        map_repository=mock_repositories["map_repository"],
        team_repository=mock_repositories["team_repository"],
        player_repository=mock_repositories["player_repository"],
    )


@pytest.fixture
def mock_inference_task():
    task_mock = MagicMock()
    task_mock.id = "fake-task-id"
    task_mock.delay.return_value = task_mock
    return task_mock


@pytest.fixture
def forecaster_service(mock_inference_task):
    return ForecasterService(inference_model_task=mock_inference_task)


# ------------------------------
# DictionaryService tests
# ------------------------------
def test_validate_game(dictionary_service):
    game = {
        "map": {"id": 1, "name": "de_dust2"},
        "begin_at": "2025-01-01T12:00:00Z",
        "players": [
            {"team": {"id": 100}, "player": {"id": i + 1, "name": f"p{i + 1}"}}
            for i in range(5)
        ]
        + [
            {"team": {"id": 200}, "player": {"id": i + 6, "name": f"p{i + 6}"}}
            for i in range(5)
        ],
        "rounds": [
            {"round": i, "winner_team": 100 if i % 2 else 200} for i in range(1, 17)
        ],
    }
    assert dictionary_service._validate_game(game) is True

    # invalid game: remove one player
    game["players"].pop()
    assert dictionary_service._validate_game(game) is False


def test_load_dictionaries_from_json(tmp_path, dictionary_service, mock_repositories):
    game_data = {
        "map": {"id": 1, "name": "de_inferno"},
        "begin_at": "2025-01-01T12:00:00Z",
        "players": [
            {
                "team": {"id": 10, "name": "Team A"},
                "player": {"id": i + 1, "name": name},
            }
            for i, name in enumerate(["Alice", "Bob", "Charlie", "David", "Eve"])
        ]
        + [
            {
                "team": {"id": 20, "name": "Team B"},
                "player": {"id": i + 6, "name": name},
            }
            for i, name in enumerate(["Frank", "Grace", "Heidi", "Ivan", "Judy"])
        ],
        "rounds": [
            {"round": i, "winner_team": 10 if i % 2 else 20} for i in range(1, 17)
        ],
    }
    file_path = tmp_path / "game.json"
    file_path.write_text(json.dumps(game_data), encoding="utf-8")

    mock_repositories["map_repository"].save.return_value = {
        "map_id": 1,
        "name": "de_inferno",
    }
    mock_repositories["team_repository"].save.return_value = {
        "team_id": 10,
        "name": "Team A",
    }
    mock_repositories["player_repository"].save.return_value = {
        "player_id": 1,
        "name": "Alice",
    }

    dictionary_service.load_dictionaries_from_json(tmp_path)
    mock_repositories["map_repository"].save.assert_called_once_with(
        map_id=1, name="de_inferno"
    )
    assert mock_repositories["team_repository"].save.call_count == 10
    assert mock_repositories["player_repository"].save.call_count == 10


def test_load_dictionaries_from_json_invalid_json(tmp_path, dictionary_service):
    file_path = tmp_path / "bad.json"
    file_path.write_text("{invalid_json}", encoding="utf-8")
    dictionary_service.load_dictionaries_from_json(tmp_path)  # Should not raise


# Search methods
def test_search_map_by_name(dictionary_service, mock_repositories):
    mock_repositories["map_repository"].search_by_name.return_value = [
        {"map_id": 1, "name": "de_dust2"}
    ]
    result = dictionary_service.search_map_by_name("dust")
    assert result[0]["name"] == "de_dust2"


def test_search_team_by_name(dictionary_service, mock_repositories):
    mock_repositories["team_repository"].search_by_name.return_value = [
        {"team_id": 100, "name": "TeamX"}
    ]
    result = dictionary_service.search_team_by_name("TeamX")
    assert result[0]["name"] == "TeamX"


def test_search_player_by_name(dictionary_service, mock_repositories):
    mock_repositories["player_repository"].search_by_name.return_value = [
        {"player_id": 1, "name": "Alice"}
    ]
    result = dictionary_service.search_player_by_name("Alice")
    assert result[0]["name"] == "Alice"


def test_get_forecast_result_pending(monkeypatch, forecaster_service):
    mock_async_result = MagicMock()
    mock_async_result.state = "PENDING"
    mock_async_result.result = None
    monkeypatch.setattr(
        "backend.services.AsyncResult", lambda forecast_id: mock_async_result
    )

    result = forecaster_service.get_forecast_result_by_id("any-id")
    assert result["status"] == "PENDING"
    assert result["forecast_id"] == "any-id"
    assert "result" not in result


def test_get_forecast_result_failure(monkeypatch, forecaster_service):
    mock_async_result = MagicMock()
    mock_async_result.state = "FAILURE"
    mock_async_result.result = None
    monkeypatch.setattr(
        "backend.services.AsyncResult", lambda forecast_id: mock_async_result
    )

    result = forecaster_service.get_forecast_result_by_id("any-id")
    assert result["status"] == "FAILURE"
    assert result["forecast_id"] == "any-id"
    assert "result" not in result


def test_get_forecast_result_success(monkeypatch, forecaster_service):
    mock_async_result = MagicMock()
    mock_async_result.state = "SUCCESS"
    mock_async_result.result = {
        "team1_id": 100,
        "team2_id": 200,
        "team1_win_probability": 0.6,
        "team2_win_probability": 0.4,
    }
    monkeypatch.setattr(
        "backend.services.AsyncResult", lambda forecast_id: mock_async_result
    )

    result = forecaster_service.get_forecast_result_by_id("any-id")
    assert result["status"] == "SUCCESS"
    assert result["forecast_id"] == "any-id"
    assert result["result"]["team1_id"] == 100
    assert result["result"]["team2_id"] == 200


def test_get_forecast_result_unknown_state(monkeypatch, forecaster_service):
    mock_async_result = MagicMock()
    mock_async_result.state = "RETRY"
    mock_async_result.result = None
    monkeypatch.setattr(
        "backend.services.AsyncResult", lambda forecast_id: mock_async_result
    )

    result = forecaster_service.get_forecast_result_by_id("any-id")
    assert result["status"] == "RETRY"
    assert result["forecast_id"] == "any-id"
    assert "result" not in result
