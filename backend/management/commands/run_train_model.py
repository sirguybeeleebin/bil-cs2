from django.core.management.base import BaseCommand

from backend.tasks import train_model_task


class Command(BaseCommand):
    help = "Запускает задачу обучения модели через Celery"

    def handle(self, *args, **options):
        result = train_model_task.delay()
        self.stdout.write(
            self.style.SUCCESS(
                f"Задача обучения модели отправлена в Celery. task_id={result.id}"
            )
        )
