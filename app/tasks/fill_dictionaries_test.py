import json
from unittest.mock import MagicMock, mock_open, patch

import pytest

from app.tasks.fill_dictionaries import make_fill_dictionaries_task


@pytest.fixture
def mock_repositories():
    map_repo = MagicMock()
    team_repo = MagicMock()
    player_repo = MagicMock()
    map_repo.upsert.return_value = True
    team_repo.upsert.return_value = True
    player_repo.upsert.return_value = True
    return map_repo, team_repo, player_repo


@pytest.fixture
def sample_game_file_content():
    return json.dumps(
        {
            "map": {"id": "map123", "name": "de_dust2"},
            "players": [
                {
                    "team": {"id": "team1", "name": "Alpha"},
                    "player": {"id": "p1", "name": "Player1"},
                },
                {
                    "team": {"id": "team2", "name": "Bravo"},
                    "player": {"id": "p2", "name": "Player2"},
                },
            ],
        }
    )


def test_fill_dictionaries_task_processes_files(
    mock_repositories, sample_game_file_content
):
    map_repo, team_repo, player_repo = mock_repositories

    task = make_fill_dictionaries_task(map_repo, team_repo, player_repo)

    # Patch os functions
    with (
        patch("os.path.exists", return_value=True),
        patch("os.listdir", return_value=["file1.json"]),
        patch("builtins.open", mock_open(read_data=sample_game_file_content)),
    ):
        result = task("fake_dir")

    assert result["status"] == "готово"
    # Check that upsert was called for map, teams, and players
    map_repo.upsert.assert_called_once_with({"map_id": "map123", "name": "de_dust2"})
    team_repo.upsert.assert_any_call({"team_id": "team1", "name": "Alpha"})
    team_repo.upsert.assert_any_call({"team_id": "team2", "name": "Bravo"})
    player_repo.upsert.assert_any_call({"player_id": "p1", "name": "Player1"})
    player_repo.upsert.assert_any_call({"player_id": "p2", "name": "Player2"})


def test_fill_dictionaries_task_handles_missing_directory(mock_repositories):
    map_repo, team_repo, player_repo = mock_repositories
    task = make_fill_dictionaries_task(map_repo, team_repo, player_repo)

    with patch("os.path.exists", return_value=False):
        result = task("missing_dir")

    assert result["status"] == "ошибка"
    assert "не существует" in result["message"]


def test_fill_dictionaries_task_handles_invalid_json(mock_repositories):
    map_repo, team_repo, player_repo = mock_repositories
    task = make_fill_dictionaries_task(map_repo, team_repo, player_repo)

    with (
        patch("os.path.exists", return_value=True),
        patch("os.listdir", return_value=["bad.json"]),
        patch("builtins.open", mock_open(read_data="not json")),
    ):
        result = task("fake_dir")

    # Should still finish and return готово
    assert result["status"] == "готово"
    # upsert should not have been called because JSON is invalid
    map_repo.upsert.assert_not_called()
    team_repo.upsert.assert_not_called()
    player_repo.upsert.assert_not_called()


def test_fill_dictionaries_task_skips_missing_keys(mock_repositories):
    map_repo, team_repo, player_repo = mock_repositories
    task = make_fill_dictionaries_task(map_repo, team_repo, player_repo)

    incomplete_data = json.dumps({"players": [{"team": {}, "player": {}}]})

    with (
        patch("os.path.exists", return_value=True),
        patch("os.listdir", return_value=["file.json"]),
        patch("builtins.open", mock_open(read_data=incomplete_data)),
    ):
        result = task("fake_dir")

    # Task should still finish
    assert result["status"] == "готово"
    # upsert should not have been called due to missing IDs
    map_repo.upsert.assert_not_called()
    team_repo.upsert.assert_not_called()
    player_repo.upsert.assert_not_called()
