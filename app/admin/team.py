from django.contrib import admin

from app.models.team import Team


@admin.register(Team)
class TeamAdmin(admin.ModelAdmin):
    list_display = ("team_id", "name", "created_at", "updated_at")
    search_fields = ("name",)
    ordering = ("name",)
