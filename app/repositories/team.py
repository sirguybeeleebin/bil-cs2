import logging
import uuid

from app.models.team import Team

log = logging.getLogger(__name__)


class TeamRepository:
    def upsert(self, data: dict) -> dict | None:
        try:
            data["team_id"] = data.get("team_id") or uuid.uuid4()
            obj, _ = Team.objects.update_or_create(
                team_id=data["team_id"], defaults=data
            )
            return obj.__dict__
        except Exception:
            log.exception("Ошибка при upsert Team")
            return None

    def search_by_name(self, name: str) -> list[dict]:
        try:
            results = Team.objects.filter(name__icontains=name)
            return [obj.__dict__ for obj in results]
        except Exception:
            log.exception("Ошибка при search_by_name Team")
            return []

    def get_by_name(self, name: str) -> dict | None:
        try:
            obj = Team.objects.get(name=name)
            return obj.__dict__
        except Team.DoesNotExist:
            return None
        except Exception:
            log.exception("Ошибка при get_by_name Team")
            return None


def make_team_repository() -> TeamRepository:
    return TeamRepository()
