from unittest.mock import Mock

import httpx

from etl.load import load_map, load_players, load_teams


# -----------------------
# Test load_map
# -----------------------
def test_load_map_calls_post():
    client = Mock(spec=httpx.Client)
    client.post.return_value = Mock()
    map_data = {"map_id": "1", "name": "Dust2"}
    headers = {"Authorization": "Bearer token"}
    load_map(map_data, client, "http://example.com/maps", headers=headers)
    client.post.assert_called_once_with(
        "http://example.com/maps", json=map_data, headers=headers
    )


def test_load_map_none_data():
    client = Mock(spec=httpx.Client)
    load_map(None, client, "http://example.com/maps")
    client.post.assert_not_called()


def test_load_map_http_error():
    client = Mock(spec=httpx.Client)
    resp = Mock()
    resp.raise_for_status.side_effect = httpx.HTTPError("error")
    client.post.return_value = resp
    load_map({"map_id": "1", "name": "Dust2"}, client, "http://example.com/maps")
    client.post.assert_called_once()


# -----------------------
# Test load_teams
# -----------------------
def test_load_teams_calls_post_for_each_team():
    client = Mock(spec=httpx.Client)
    client.post.return_value = Mock()
    teams_data = [{"team_id": "1", "name": "TeamA"}, {"team_id": "2", "name": "TeamB"}]
    headers = {"Authorization": "Bearer token"}
    load_teams(teams_data, client, "http://example.com/teams", headers=headers)
    assert client.post.call_count == 2


def test_load_teams_empty_list():
    client = Mock(spec=httpx.Client)
    load_teams([], client, "http://example.com/teams")
    client.post.assert_not_called()


def test_load_teams_http_error_continues():
    client = Mock(spec=httpx.Client)
    resp = Mock()
    resp.raise_for_status.side_effect = httpx.HTTPError("error")
    client.post.return_value = resp
    teams_data = [{"team_id": "1", "name": "TeamA"}, {"team_id": "2", "name": "TeamB"}]
    load_teams(teams_data, client, "http://example.com/teams")
    assert client.post.call_count == 2  # should attempt all even if errors


# -----------------------
# Test load_players
# -----------------------
def test_load_players_calls_post_for_each_player():
    client = Mock(spec=httpx.Client)
    client.post.return_value = Mock()
    players_data = [
        {"player_id": "101", "name": "Alice"},
        {"player_id": "102", "name": "Bob"},
    ]
    headers = {"Authorization": "Bearer token"}
    load_players(players_data, client, "http://example.com/players", headers=headers)
    assert client.post.call_count == 2


def test_load_players_empty_list():
    client = Mock(spec=httpx.Client)
    load_players([], client, "http://example.com/players")
    client.post.assert_not_called()


def test_load_players_http_error_continues():
    client = Mock(spec=httpx.Client)
    resp = Mock()
    resp.raise_for_status.side_effect = httpx.HTTPError("error")
    client.post.return_value = resp
    players_data = [
        {"player_id": "101", "name": "Alice"},
        {"player_id": "102", "name": "Bob"},
    ]
    load_players(players_data, client, "http://example.com/players")
    assert client.post.call_count == 2
