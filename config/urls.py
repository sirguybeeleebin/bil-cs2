from channels.layers import get_channel_layer
from django.contrib import admin
from django.urls import include, path
from drf_spectacular.views import (
    SpectacularAPIView,
    SpectacularRedocView,
    SpectacularSwaggerView,
)

from app.handlers.forecast import make_forecast_handler
from app.handlers.map import (
    make_map_get_by_name_handler,
    make_map_search_by_name_handler,
)
from app.handlers.player import (
    make_player_get_by_name_handler,
    make_player_search_by_name_handler,
)
from app.handlers.team import (
    make_team_get_by_name_handler,
    make_team_search_by_name_handler,
)
from app.repositories.map import make_map_repository
from app.repositories.ml_forecast import make_ml_forecast_repository
from app.repositories.player import make_player_repository
from app.repositories.team import make_team_repository
from app.tasks.ml_pipeline_inference import make_ml_inference_task

ml_forecast_repo = make_ml_forecast_repository()
channel_layer = get_channel_layer()

ml_inference_task = make_ml_inference_task(
    ml_forecast_repository=ml_forecast_repo,
    channel_layer=channel_layer,
)

forecast_handler = make_forecast_handler(
    ml_forecast_repository=ml_forecast_repo,
    run_inference_task=ml_inference_task,
)

map_repo = make_map_repository()
map_get_by_name_handler = make_map_get_by_name_handler(map_repository=map_repo)
map_search_by_name_handler = make_map_search_by_name_handler(map_repository=map_repo)

team_repo = make_team_repository()
team_get_by_name_handler = make_team_get_by_name_handler(team_repository=team_repo)
team_search_by_name_handler = make_team_search_by_name_handler(
    team_repository=team_repo
)

player_repo = make_player_repository()
player_get_by_name_handler = make_player_get_by_name_handler(
    player_repository=player_repo
)
player_search_by_name_handler = make_player_search_by_name_handler(
    player_repository=player_repo
)

v1_urlpatterns = [
    path("forecast/", forecast_handler.as_view(), name="forecast"),
    path(
        "map/get_by_name/<str:name>/",
        map_get_by_name_handler.as_view(),
        name="map-get-by-name",
    ),
    path(
        "map/search_by_name/<str:name>/",
        map_search_by_name_handler.as_view(),
        name="map-search-by-name",
    ),
    path(
        "team/get_by_name/<str:name>/",
        team_get_by_name_handler.as_view(),
        name="team-get-by-name",
    ),
    path(
        "team/search_by_name/<str:name>/",
        team_search_by_name_handler.as_view(),
        name="team-search-by-name",
    ),
    path(
        "player/get_by_name/<str:name>/",
        player_get_by_name_handler.as_view(),
        name="player-get-by-name",
    ),
    path(
        "player/search_by_name/<str:name>/",
        player_search_by_name_handler.as_view(),
        name="player-search-by-name",
    ),
]

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/v1/", include((v1_urlpatterns, "v1"), namespace="v1")),
    path("api/schema/", SpectacularAPIView.as_view(), name="schema"),
    path(
        "api/docs/swagger/",
        SpectacularSwaggerView.as_view(url_name="schema"),
        name="swagger-ui",
    ),
    path(
        "api/docs/redoc/",
        SpectacularRedocView.as_view(url_name="schema"),
        name="redoc",
    ),
]
