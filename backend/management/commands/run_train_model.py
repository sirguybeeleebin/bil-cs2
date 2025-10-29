from django.core.management.base import BaseCommand

from backend.tasks import train_model_task


class Command(BaseCommand):
    help = "Запуск тренировки ML модели"

    def handle(self, *args, **options):
        result = train_model_task.apply_async()
        self.stdout.write(f"Запущена тренировка модели, task_id={result.id}")
