from django.apps import AppConfig as DjangoAppConfig
from internal.handlers import MapHandler, PlayerHandler, TeamHandler, Team1WinProbabilityHandler
from internal.services import CS2Service, MLService
from internal.repositories import MapRepository, TeamRepository, PlayerRepository


class AppConfig(DjangoAppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "app"

    def ready(self):
        # --- Инициализация репозиториев ---
        map_repository = MapRepository()
        team_repository = TeamRepository()
        player_repository = PlayerRepository()

        # --- Инициализация сервисов с DI ---
        cs2_service = CS2Service(
            map_repository=map_repository,
            team_repository=team_repository,
            player_repository=player_repository,
        )

        ml_service = MLService()

        # --- Регистрация обработчиков с DI и сохранение в атрибутах ---
        self.map_handler = MapHandler.as_view(cs2_service=cs2_service)
        self.player_handler = PlayerHandler.as_view(cs2_service=cs2_service)
        self.team_handler = TeamHandler.as_view(cs2_service=cs2_service)
        self.team1_win_probability_handler = Team1WinProbabilityHandler.as_view(ml_service=ml_service)

        # --- Сохраняем сервисы и репозитории для доступа из других частей приложения ---
        self.cs2_service = cs2_service
        self.ml_service = ml_service
        self.map_repository = map_repository
        self.team_repository = team_repository
        self.player_repository = player_repository
