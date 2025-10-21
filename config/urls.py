# config/urls.py
from django.contrib import admin
from django.urls import path, include
from django.apps import apps

# internal_config = apps.get_app_config("app")

# ml_api_urls = [
#     path("maps/<str:name>/", internal_config.get_map_by_name_view),
#     path("teams/<str:name>/", internal_config.get_team_by_name_view),
#     path("players/<str:name>/", internal_config.get_player_by_name_view),
#     path("team1_win_probability/", internal_config.get_team1_win_probability_view),
# ]

urlpatterns = [
    path("admin/", admin.site.urls),
    # path("api/v1/", include((ml_api_urls, "ml_api"), namespace="ml_api")),
]
