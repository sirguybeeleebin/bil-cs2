from etl.transform import transform_map, transform_player, transform_team, validate_game

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


# -----------------------
# Sample correct game
# -----------------------
valid_game = {
    "map": {"id": 1, "name": "Dust2"},
    "begin_at": "2025-01-01T12:00:00Z",
    "players": [
        {"player": {"id": 101}, "team": {"id": 1}},
        {"player": {"id": 102}, "team": {"id": 1}},
        {"player": {"id": 103}, "team": {"id": 1}},
        {"player": {"id": 104}, "team": {"id": 1}},
        {"player": {"id": 105}, "team": {"id": 1}},
        {"player": {"id": 201}, "team": {"id": 2}},
        {"player": {"id": 202}, "team": {"id": 2}},
        {"player": {"id": 203}, "team": {"id": 2}},
        {"player": {"id": 204}, "team": {"id": 2}},
        {"player": {"id": 205}, "team": {"id": 2}},
    ],
    "rounds": [
        {
            "round": i + 1,
            "ct": 1 if i % 2 == 0 else 2,
            "terrorists": 2 if i % 2 == 0 else 1,
            "winner_team": 1 if i % 2 == 0 else 2,
        }
        for i in range(16)
    ],
}


# -----------------------
# Tests
# -----------------------
def test_validate_game_correct():
    assert validate_game(valid_game) is True


def test_validate_game_wrong_team_count():
    game = valid_game.copy()
    game["players"] = game["players"][:5]  # Only one team
    assert validate_game(game) is False


def test_validate_game_wrong_player_count():
    game = valid_game.copy()
    # Make team 1 have only 4 players
    game["players"] = game["players"][:4] + game["players"][5:]
    assert validate_game(game) is False


def test_validate_game_invalid_round_ct():
    game = valid_game.copy()
    game["rounds"][0]["ct"] = 3  # Invalid team id
    assert validate_game(game) is False


def test_validate_game_invalid_round_terrorists():
    game = valid_game.copy()
    game["rounds"][0]["terrorists"] = 3  # Invalid team id
    assert validate_game(game) is False


def test_validate_game_invalid_round_winner():
    game = valid_game.copy()
    game["rounds"][0]["winner_team"] = 3  # Invalid team id
    assert validate_game(game) is False


def test_validate_game_min_round_less_than_16():
    game = valid_game.copy()
    game["rounds"] = game["rounds"][:10]  # Only 10 rounds
    assert validate_game(game) is False


def test_validate_game_missing_keys():
    game = {}
    assert validate_game(game) is None
