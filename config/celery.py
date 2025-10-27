import logging
import os

# ruff: noqa: E402
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django  # noqa: E402

django.setup()  # noqa: E402

from celery import Celery
from celery.schedules import crontab
from celery.signals import worker_ready

log = logging.getLogger(__name__)

celery = Celery("config")
celery.config_from_object("django.conf:settings", namespace="CELERY")
celery.autodiscover_tasks()
celery.conf.timezone = "UTC"

# ruff: noqa: E402
from app.ml.train_model import train_model
from app.repositories.map import make_map_repository
from app.repositories.ml_pipeline import make_ml_pipeline_repository
from app.repositories.ml_pipeline_metrics import make_ml_pipeline_metrics_repository
from app.repositories.player import make_player_repository
from app.repositories.team import make_team_repository
from app.tasks.fill_dictionaries import make_fill_dictionaries_task
from app.tasks.ml_pipeline import make_ml_pipeline_task
from config.settings import ML_PIPELINE_SETTINGS, PATH_TO_GAMES_RAW_DIR

map_repository = make_map_repository()
team_repository = make_team_repository()
player_repository = make_player_repository()
ml_pipeline_repository = make_ml_pipeline_repository()
ml_pipeline_metrics_repository = make_ml_pipeline_metrics_repository()

fill_dictionaries_task = make_fill_dictionaries_task(
    map_repository=map_repository,
    team_repository=team_repository,
    player_repository=player_repository,
)

ml_pipeline_task = make_ml_pipeline_task(
    ml_pipeline_repository=ml_pipeline_repository,
    ml_pipeline_metrics_repository=ml_pipeline_metrics_repository,
    train_model_fn=train_model,
)

celery.conf.beat_schedule = {
    "fill-dictionaries-every-hour": {
        "task": fill_dictionaries_task.name,
        "schedule": 3600.0,
        "args": [str(PATH_TO_GAMES_RAW_DIR)],
    },
    "run-ml-pipeline-daily": {
        "task": ml_pipeline_task.name,
        "schedule": crontab(minute=0, hour=0),
        "kwargs": {
            "path_to_games_raw_dir": str(PATH_TO_GAMES_RAW_DIR),
            "test_size": ML_PIPELINE_SETTINGS["TEST_SIZE"],
            "n_splits": ML_PIPELINE_SETTINGS["N_SPLITS"],
            "n_iters": ML_PIPELINE_SETTINGS["N_ITERS"],
            "random_state": ML_PIPELINE_SETTINGS["RANDOM_STATE"],
        },
    },
}


@worker_ready.connect
def at_start(sender=None, **kwargs):
    log.info("Worker ready — запуск стартовых задач")
    fill_dictionaries_task.delay(str(PATH_TO_GAMES_RAW_DIR))
    ml_pipeline_task.delay(
        path_to_games_raw_dir=str(PATH_TO_GAMES_RAW_DIR),
        test_size=ML_PIPELINE_SETTINGS["TEST_SIZE"],
        n_splits=ML_PIPELINE_SETTINGS["N_SPLITS"],
        n_iters=ML_PIPELINE_SETTINGS["N_ITERS"],
        random_state=ML_PIPELINE_SETTINGS["RANDOM_STATE"],
    )


__all__ = ("celery",)
