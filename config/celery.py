import os
from celery import Celery
from celery.signals import worker_ready
from celery.schedules import crontab
from config.settings import ML_PIPELINE_SETTINGS, PATH_TO_GAMES_RAW_DIR

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

app = Celery("config")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()

app.conf.beat_schedule = {
    "fill-dictionaries-every-hour": {
        "task": "app.tasks.fill_dictionaries_task",
        "schedule": 3600.0,  
        "args": [str(PATH_TO_GAMES_RAW_DIR)],
    },
    "run-ml-pipeline-daily": {
        "task": "app.tasks.run_ml_pipeline_task",
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

app.conf.timezone = "UTC"


@worker_ready.connect
def at_start(sender=None, **kwargs):
    from app.tasks import fill_dictionaries_task, run_ml_pipeline_task
    fill_dictionaries_task.delay(str(PATH_TO_GAMES_RAW_DIR))
    run_ml_pipeline_task.delay(
        path_to_games_raw_dir=str(PATH_TO_GAMES_RAW_DIR),
        test_size=ML_PIPELINE_SETTINGS["TEST_SIZE"],
        n_splits=ML_PIPELINE_SETTINGS["N_SPLITS"],
        n_iters=ML_PIPELINE_SETTINGS["N_ITERS"],
        random_state=ML_PIPELINE_SETTINGS["RANDOM_STATE"],
    )
