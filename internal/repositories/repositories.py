import logging
from internal.models import Map, Player, Team

log = logging.getLogger(__name__)


class MapRepository:
    def upsert(self, data: dict) -> dict | None:
        try:
            map_id = data.pop("map_id")
            map_obj, created = Map.objects.update_or_create(
                map_id=map_id,
                defaults=data
            )
            action = "создан" if created else "обновлён"
            result = map_obj.__dict__
            log.info(f"Map {action}: {result}")
            return result
        except Exception as e:
            log.error(f"Ошибка при upsert Map: {data}, {e}")
            return None


class TeamRepository:
    def upsert(self, data: dict) -> dict | None:
        try:
            team_id = data.pop("team_id")
            team_obj, created = Team.objects.update_or_create(
                team_id=team_id,
                defaults=data
            )
            result = team_obj.__dict__
            action = "создан" if created else "обновлён"            
            log.info(f"Team {action}: {result}")
            return result
        except Exception as e:
            log.error(f"Ошибка при upsert Team: {data}, {e}")
            return None


class PlayerRepository:
    def upsert(self, data: dict) -> dict | None:
        try:
            player_id = data.pop("player_id")
            player_obj, created = Player.objects.update_or_create(
                player_id=player_id,
                defaults=data
            )
            result = player_obj.__dict__
            action = "создан" if created else "обновлён"            
            log.info(f"Player {action}: {result}")
            return result
        except Exception as e:
            log.error(f"Ошибка при upsert Player: {data}, {e}")
            return None
