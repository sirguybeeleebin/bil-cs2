import logging
import uuid

from app.models.player import Player

log = logging.getLogger(__name__)


class PlayerRepository:
    def upsert(self, data: dict) -> dict | None:
        try:
            data["player_id"] = data.get("player_id") or uuid.uuid4()
            obj, _ = Player.objects.update_or_create(
                player_id=data["player_id"], defaults=data
            )
            return obj.__dict__
        except Exception:
            log.exception("Ошибка при upsert Player")
            return None

    def search_by_name(self, name: str) -> list[dict]:
        try:
            results = Player.objects.filter(name__icontains=name)
            return [obj.__dict__ for obj in results]
        except Exception:
            log.exception("Ошибка при search_by_name Player")
            return []

    def get_by_name(self, name: str) -> dict | None:
        try:
            obj = Player.objects.get(name=name)
            return obj.__dict__
        except Player.DoesNotExist:
            return None
        except Exception:
            log.exception("Ошибка при get_by_name Player")
            return None


def make_player_repository() -> PlayerRepository:
    return PlayerRepository()
