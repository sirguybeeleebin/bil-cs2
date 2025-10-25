# tasks_factory.py
import os
import json
import uuid
from pathlib import Path
import joblib
import redis
from celery import Celery

def make_train_and_publish_task(
    celery_app: Celery,
    ml_results_path: str,
    redis_client: redis.Redis,
    redis_channel: str,
    run_ml_pipeline_func
):
    Path(ml_results_path).mkdir(parents=True, exist_ok=True)

    @celery_app.task
    def train_and_publish_task(
        data_path: str = "data/games_raw",
        test_size: int = 100,
        n_splits: int = 10,
        n_iters: int = 10,
        random_state: int = 42,
    ):
        pipeline, metrics = run_ml_pipeline_func(
            data_path=data_path,
            test_size=test_size,
            n_splits=n_splits,
            n_iters=n_iters,
            random_state=random_state
        )

        pipeline_uuid = str(uuid.uuid4())
        pipeline_path = Path(ml_results_path) / f"{pipeline_uuid}.joblib"
        metrics_path = Path(ml_results_path) / f"{pipeline_uuid}.json"

        joblib.dump(pipeline, pipeline_path)
        with metrics_path.open("w") as f:
            json.dump(metrics, f, indent=4)

        message = {
            "pipeline_id": pipeline_uuid,
            "model_file": str(pipeline_path),
            "metrics_file": str(metrics_path),
            "metrics": metrics,
        }

        redis_client.publish(redis_channel, json.dumps(message))
        print(f"Pipeline {pipeline_uuid} published to Redis")
        return message

    return train_and_publish_task
