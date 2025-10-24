from etl.transform import transform_map, transform_player, transform_team

# -----------------------
# Sample game data
# -----------------------
sample_game = {
    "map": {"id": 1, "name": "Dust2"},
    "players": [
        {"player": {"id": 101, "name": "Alice"}, "team": {"id": 1, "name": "TeamA"}},
        {"player": {"id": 101, "name": "Alice"}, "team": {"id": 1, "name": "TeamA"}},
        {"player": {"id": 102, "name": "Bob"}, "team": {"id": 2, "name": "TeamB"}},
        {"player": {"id": 103, "name": "Charlie"}, "team": {"id": 1, "name": "TeamA"}},
    ],
}


# -----------------------
# Tests for transform_map
# -----------------------
def test_transform_map_valid():
    result = transform_map(sample_game)
    assert result == {"map_id": "1", "name": "Dust2"}


def test_transform_map_missing_map_key():
    game = {"players": []}
    result = transform_map(game)
    assert result is None


def test_transform_map_empty_name():
    game = {"map": {"id": 2, "name": None}, "players": []}
    result = transform_map(game)
    assert result == {"map_id": "2", "name": ""}


# -----------------------
# Tests for transform_team
# -----------------------
def test_transform_team_valid():
    result = transform_team(sample_game)
    expected = [
        {"team_id": "1", "name": "TeamA"},
        {"team_id": "2", "name": "TeamB"},
    ]
    assert result == expected


def test_transform_team_missing_players():
    game = {"map": {"id": 1, "name": "Dust2"}}
    result = transform_team(game)
    assert result is None


def test_transform_team_empty_name():
    game = {
        "map": {"id": 1, "name": "Dust2"},
        "players": [
            {"player": {"id": 101, "name": "Alice"}, "team": {"id": 1, "name": None}}
        ],
    }
    result = transform_team(game)
    assert result == [{"team_id": "1", "name": ""}]


# -----------------------
# Tests for transform_player
# -----------------------
def test_transform_player_valid():
    result = transform_player(sample_game)
    expected = [
        {"player_id": "101", "name": "Alice"},
        {"player_id": "102", "name": "Bob"},
        {"player_id": "103", "name": "Charlie"},
    ]
    assert result == expected


def test_transform_player_missing_players():
    game = {"map": {"id": 1, "name": "Dust2"}}
    result = transform_player(game)
    assert result is None


def test_transform_player_empty_name():
    game = {
        "map": {"id": 1, "name": "Dust2"},
        "players": [
            {"player": {"id": 101, "name": None}, "team": {"id": 1, "name": "TeamA"}}
        ],
    }
    result = transform_player(game)
    assert result == [{"player_id": "101", "name": ""}]
