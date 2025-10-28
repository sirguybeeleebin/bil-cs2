from __future__ import absolute_import, unicode_literals

import json
import os

from celery import Celery, chain

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

app = Celery("config")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()


@app.on_after_finalize.connect
def setup_tasks(sender, **kwargs):
    from django_celery_beat.models import IntervalSchedule, PeriodicTask

    from backend.di import (
        map_repo,
        ml_result_metrics_repo,
        ml_result_repo,
        player_repo,
        team_repo,
    )
    from backend.tasks import (
        make_load_dictionaries_task,
        make_train_model_task,
        make_update_dictionaries_task,
    )
    from train_model.train_model import train_model
    from update_dictionaries.update_dictionaries import update_dictionaries

    # --- Создание задач через фабрики ---
    update_dicts_task = make_update_dictionaries_task(
        update_dictionaries_func=update_dictionaries
    )
    load_dicts_task = make_load_dictionaries_task(
        map_repository=map_repo,
        team_repository=team_repo,
        player_repository=player_repo,
    )
    train_model_task = make_train_model_task(
        train_model_func=train_model,
        ml_result_repository=ml_result_repo,
        ml_result_metrics_repository=ml_result_metrics_repo,
    )

    # --- Цепочка обновления и загрузки словарей ---
    @app.task(name="backend.tasks.chain_update_and_load")
    def chain_update_and_load_task():
        return chain(update_dicts_task.s(), load_dicts_task.s()).apply_async()

    # --- Регистрируем расписания в БД ---
    hourly, _ = IntervalSchedule.objects.get_or_create(
        every=3600,
        period=IntervalSchedule.SECONDS,
    )
    daily, _ = IntervalSchedule.objects.get_or_create(
        every=86400,
        period=IntervalSchedule.SECONDS,
    )

    # --- Добавляем/обновляем задачи в Django admin ---
    PeriodicTask.objects.update_or_create(
        name="update-dictionaries-every-hour",
        defaults={
            "task": "backend.tasks.chain_update_and_load",
            "interval": hourly,
            "args": json.dumps([]),
        },
    )

    PeriodicTask.objects.update_or_create(
        name="train-model-daily-midnight",
        defaults={
            "task": train_model_task.name,
            "interval": daily,
            "args": json.dumps([]),
        },
    )


if os.environ.get("RUN_MAIN") is None and os.environ.get("CELERY_WORKER") == "1":
    setup_tasks()
