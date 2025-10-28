from django.contrib import admin
from django.urls import path
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

from backend.init_handlers import (
    forecast_handler,
    map_search_handler,
    player_search_handler,
    team_search_handler,
)

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/v1/token/", TokenObtainPairView.as_view(), name="token_obtain_pair"),
    path("api/v1/token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    path("api/v1/schema/", SpectacularAPIView.as_view(), name="schema"),
    path(
        "api/v1/docs/",
        SpectacularSwaggerView.as_view(url_name="schema"),
        name="swagger-ui",
    ),
    path("api/v1/maps/", map_search_handler.as_view(), name="map-search"),
    path("api/v1/teams/", team_search_handler.as_view(), name="team-search"),
    path("api/v1/players/", player_search_handler.as_view(), name="player-search"),
    path("api/v1/forecast/", forecast_handler.as_view(), name="forecast"),
]
