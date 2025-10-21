import logging
from django.apps import AppConfig as DjangoAppConfig

log = logging.getLogger(__name__)

class AppConfig(DjangoAppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "app"

    def ready(self):
        from app import handlers, tasks
        from app.repositories.map import make_map_repository
        from app.repositories.team import make_team_repository
        from app.repositories.player import make_player_repository
        # from app.services import make_etl_service, make_ml_service, make_cs2_service
        
        self.map_repository = make_map_repository()
        self.team_repository = make_team_repository()
        self.player_repository = make_player_repository()
        
        # self.cs2_service = make_cs2_service(
        #     self.map_repository, self.team_repository, self.player_repository
        # )
        # self.ml_service = make_ml_service()
        # self.etl_service = make_etl_service(
        #     self.map_repository, self.team_repository, self.player_repository
        # )
        
        # self.get_map_by_name_handler = handlers.make_get_map_by_name_handler(
        #     self.cs2_service
        # )
        # self.get_team_by_name_handler = handlers.make_get_team_by_name_handler(
        #     self.cs2_service
        # )
        # self.get_player_by_name_handler = handlers.make_get_player_by_name_handler(
        #     self.cs2_service
        # )
        # self.get_team1_win_probability_handler = handlers.make_get_team1_win_probability_handler(
        #     self.ml_service
        # )
        
        # self.run_etl_task = tasks.make_run_etl_task(self.etl_service)
        # self.run_ml_task = tasks.make_run_ml_task(self.ml_service)

        log.info(
            "DI инициализировано при старте Django: репозитории, сервисы, handlers, Celery задачи"
        )
