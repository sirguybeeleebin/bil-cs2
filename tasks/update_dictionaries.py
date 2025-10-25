# tasks_update.py
import os
import json
from pathlib import Path
from celery import Celery

def make_update_dictionaries_task(
    celery_app: Celery, 
    redis_client, 
    redis_channel: str,
    path_game_raw_data_dir: Path,
):
    @celery_app.task
    def update_dictionaries():
        if not path_game_raw_data_dir.exists() or not path_game_raw_data_dir.is_dir():
            print(f"Data directory {path_game_raw_data_dir} does not exist or is not a directory")
            return {"status": "failed", "reason": "invalid directory"}

        published_count = 0

        # Iterate over all JSON files in the directory
        for file_path in path_game_raw_data_dir.glob("*.json"):
            try:
                with file_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                message = {
                    "update_id": f"{file_path.stem}_{os.urandom(4).hex()}",
                    "file": str(file_path),
                    "data": data
                }
                redis_client.publish(redis_channel, json.dumps(message))
                published_count += 1
                print(f"Published {file_path} to Redis: {message['update_id']}")
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

        return {"status": "ok", "published_files": published_count}

    return update_dictionaries
