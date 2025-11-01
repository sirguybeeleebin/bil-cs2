from django.core.management.base import BaseCommand

from backend.tasks import update_dictionaries_task


class Command(BaseCommand):
    help = "Запускает задачу обновления словарей через Celery"

    def handle(self, *args, **options):
        result = update_dictionaries_task.delay()
        self.stdout.write(
            self.style.SUCCESS(
                f"Задача обновления словарей отправлена в Celery. task_id={result.id}"
            )
        )
