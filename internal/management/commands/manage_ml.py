# app/management/commands/manage_ml.py
from django.core.management.base import BaseCommand
from app.tasks import run_ml_task
from celery.result import AsyncResult
from django.core.cache import cache
import logging

log = logging.getLogger(__name__)

class Command(BaseCommand):
    help = "Start or stop ML pipeline. Use --start or --stop"

    def add_arguments(self, parser):
        parser.add_argument(
            '--start',
            action='store_true',
            help='Start the ML pipeline'
        )
        parser.add_argument(
            '--stop',
            action='store_true',
            help='Stop the ML pipeline'
        )

    def handle(self, *args, **options):
        start = options['start']
        stop = options['stop']

        if start:
            log.info("Starting ML task via Celery...")
            result = run_ml_task.delay()
            cache.set("ml_task_id", result.id)
            self.stdout.write(self.style.SUCCESS(f"ML task started. Task ID: {result.id}"))

        elif stop:
            task_id = cache.get("ml_task_id")
            if not task_id:
                self.stdout.write(self.style.WARNING("No ML task ID found"))
                return
            AsyncResult(task_id).revoke(terminate=True, signal='SIGKILL')
            cache.delete("ml_task_id")
            self.stdout.write(self.style.SUCCESS(f"ML task {task_id} stopped"))

        else:
            self.stdout.write(self.style.WARNING("Please provide --start or --stop"))
