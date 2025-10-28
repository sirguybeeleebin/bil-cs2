from django.core.management.base import BaseCommand

from backend.di import ml_result_metrics_repo, ml_result_repo
from backend.tasks import make_train_model_task
from train_model.train_model import train_model


class Command(BaseCommand):
    help = "Запуск тренировки ML модели"

    def handle(self, *args, **options):
        task = make_train_model_task(
            train_model_func=train_model,
            ml_result_repository=ml_result_repo,
            ml_result_metrics_repository=ml_result_metrics_repo,
        )
        result = task.apply_async()
        self.stdout.write(f"Запущена тренировка модели, task_id={result.id}")
