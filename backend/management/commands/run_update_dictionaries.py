from celery import chain
from django.core.management.base import BaseCommand

from backend.di import map_repo, player_repo, team_repo
from backend.tasks import make_load_dictionaries_task, make_update_dictionaries_task
from update_dictionaries.update_dictionaries import update_dictionaries


class Command(BaseCommand):
    help = "Запуск цепочки обновления словарей и загрузки"

    def handle(self, *args, **options):
        update_task = make_update_dictionaries_task(
            update_dictionaries_func=update_dictionaries
        )
        load_task = make_load_dictionaries_task(
            map_repository=map_repo,
            team_repository=team_repo,
            player_repository=player_repo,
        )

        result = chain(update_task.s(), load_task.s()).apply_async()
        self.stdout.write(f"Запущена цепочка update + load, task_id={result.id}")
