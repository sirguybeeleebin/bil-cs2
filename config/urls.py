# config/urls.py
from django.contrib import admin
from django.urls import path, include
from django.apps import apps

internal = apps.get_app_config("internal")

ml_api_urls = [
    path("team/<str:team_name>", internal.team_handler, name="get_team_by_name"),
    path("player/<str:player_name>", internal.player_handler, name="get_player_by_name"),
    path("map/<str:map_name>", internal.map_handler, name="get_map_by_name"),
    path("team1-win-probability/", internal.team1_win_probability_handler, name="get_team1_win_probability"),
]

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/v1/", include((ml_api_urls, "ml_api"))),
]
