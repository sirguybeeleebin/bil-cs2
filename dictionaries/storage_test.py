import json
import tempfile
from pathlib import Path

from dictionaries.storage import save_json


def test_save_json_creates_file_and_content():
    data = {"key": "value", "number": 42}

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        save_json(path, data)
        assert path.exists() and path.is_file()
        with path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == data
