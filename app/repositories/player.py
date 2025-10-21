import logging
from internal.models import Player

log = logging.getLogger(__name__)

class PlayerRepository:
    def upsert(self, data: dict) -> dict | None:
        try:
            player_id = data.pop("player_id")
            player_obj, created = Player.objects.update_or_create(
                player_id=player_id,
                defaults=data
            )
            action = "создан" if created else "обновлён"
            log.info(f"Player {action}: {player_obj.__dict__}")
            return player_obj.__dict__
        except Exception as e:
            log.error(f"Ошибка при upsert Player: {data}, {e}")
            return None

    def get_by_name(self, name: str) -> dict | None:
        try:
            obj = Player.objects.filter(name__iexact=name).first()
            if not obj:
                log.info(f"Игрок с именем '{name}' не найден")
                return None
            log.info(f"Найден игрок: {obj.__dict__}")
            return obj.__dict__
        except Exception as e:
            log.error(f"Ошибка при поиске игрока '{name}': {e}")
            return None

def make_player_repository() -> PlayerRepository:
    return PlayerRepository()
