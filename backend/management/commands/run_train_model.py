from django.core.management.base import BaseCommand

from backend.di import ml_result_metrics_repo, ml_result_repo
from backend.tasks import make_train_model_task
from train_model.train_model import train_model


class Command(BaseCommand):
    help = "Запуск тренировки ML модели"

    def handle(self, *args, **options):
        # Создаём задачу через фабрику с явным именем
        task = make_train_model_task(
            train_model_func=train_model,
            ml_pipeline_repository=ml_result_repo,
            ml_pipeline_metrics_repository=ml_result_metrics_repo,
<<<<<<< HEAD
            task_name="backend.tasks.train_model_task",
=======
>>>>>>> origin/main
        )
        result = task.apply_async()
        self.stdout.write(f"Запущена тренировка модели, task_id={result.id}")
