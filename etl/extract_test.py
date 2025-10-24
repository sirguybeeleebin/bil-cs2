import json

from etl.extract import generate_game_raw


def create_json_file(tmp_path, filename, data):
    file_path = tmp_path / filename
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return file_path


def test_generate_game_raw_single_file(tmp_path):
    data = [
        {"map": {"id": 1, "name": "Dust2"}, "players": []},
        {"map": {"id": 2, "name": "Inferno"}, "players": []},
    ]
    create_json_file(tmp_path, "games.json", data)

    results = list(generate_game_raw(str(tmp_path)))
    assert results == data


def test_generate_game_raw_multiple_files(tmp_path):
    data1 = [{"map": {"id": 1, "name": "Dust2"}, "players": []}]
    data2 = [{"map": {"id": 2, "name": "Inferno"}, "players": []}]

    create_json_file(tmp_path, "games1.json", data1)
    create_json_file(tmp_path, "games2.json", data2)

    results = list(generate_game_raw(str(tmp_path)))
    assert results == data1 + data2


def test_generate_game_raw_invalid_json(tmp_path):
    # Create a valid file and an invalid JSON file
    valid_data = [{"map": {"id": 1, "name": "Dust2"}, "players": []}]
    create_json_file(tmp_path, "valid.json", valid_data)

    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text("{invalid_json", encoding="utf-8")

    results = list(generate_game_raw(str(tmp_path)))
    assert results == valid_data


def test_generate_game_raw_empty_directory(tmp_path):
    results = list(generate_game_raw(str(tmp_path)))
    assert results == []
