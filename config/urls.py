from django.contrib import admin
from django.urls import path
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

from backend.handlers import (
    ForecastHandler,
    ForecastResultHandler,
    MapHandler,
    MLMetricsHandler,
    PlayerHandler,
    RegisterHandler,
    TeamHandler,
)

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/v1/register/", RegisterHandler.as_view(), name="register"),
    path("api/v1/token/", TokenObtainPairView.as_view(), name="token_obtain_pair"),
    path("api/v1/token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    path("api/v1/maps/", MapHandler.as_view(), name="maps"),
    path("api/v1/teams/", TeamHandler.as_view(), name="teams"),
    path("api/v1/players/", PlayerHandler.as_view(), name="players"),
    path("api/v1/forecast/", ForecastHandler.as_view(), name="forecast"),
    path(
        "api/v1/forecast/<str:forecast_id>/",
        ForecastResultHandler.as_view(),
        name="forecast_result",
    ),
    path("api/v1/metrics/", MLMetricsHandler.as_view(), name="ml_metrics_latest"),
]
