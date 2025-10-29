from django.core.management.base import BaseCommand

from backend.tasks import update_and_load_dictionaries_task


class Command(BaseCommand):
    help = "Запуск цепочки обновления словарей и загрузки"

    def handle(self, *args, **options):
        result = update_and_load_dictionaries_task.apply_async()
        self.stdout.write(f"Запущена цепочка update + load, task_id={result.id}")
