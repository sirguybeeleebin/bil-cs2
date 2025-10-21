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

    def get_by_name(self, name: str) -> dict | None:
        try:
            obj = Map.objects.filter(name__iexact=name).first()
            if not obj:
                log.info(f"Карта с именем '{name}' не найдена")
                return None
            log.info(f"Найдена карта: {obj.__dict__}")
            return obj.__dict__
        except Exception as e:
            log.error(f"Ошибка при поиске карты '{name}': {e}")
            return None


class TeamRepository:
    def upsert(self, data: dict) -> dict | None:
        try:
            team_id = data.pop("team_id")
            team_obj, created = Team.objects.update_or_create(
                team_id=team_id,
                defaults=data
            )
            action = "создан" if created else "обновлён"            
            log.info(f"Team {action}: {team_obj.__dict__}")
            return team_obj.__dict__
        except Exception as e:
            log.error(f"Ошибка при upsert Team: {data}, {e}")
            return None

    def get_by_name(self, name: str) -> dict | None:
        try:
            obj = Team.objects.filter(name__iexact=name).first()
            if not obj:
                log.info(f"Команда с именем '{name}' не найдена")
                return None
            log.info(f"Найдена команда: {obj.__dict__}")
            return obj.__dict__
        except Exception as e:
            log.error(f"Ошибка при поиске команды '{name}': {e}")
            return None


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


def make_map_repository() -> MapRepository:
    return MapRepository()


def make_team_repository() -> TeamRepository:
    return TeamRepository()


def make_player_repository() -> PlayerRepository:
    return PlayerRepository()
