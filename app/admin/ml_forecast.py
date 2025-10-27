from django.contrib import admin

from app.models.ml_forecast import MLForecast


@admin.register(MLForecast)
class MLForecastAdmin(admin.ModelAdmin):
    list_display = (
        "ml_forecast_id",
        "map",
        "team1",
        "team2",
        "team1_win_probability",
        "status",
        "ml_pipeline",
        "created_at",
        "updated_at",
    )
    search_fields = ("map__name", "team1__name", "team2__name")
    list_filter = ("status", "ml_pipeline")
    filter_horizontal = ("team1_players", "team2_players")
    ordering = ("created_at",)
