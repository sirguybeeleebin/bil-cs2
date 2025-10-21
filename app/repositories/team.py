import logging
from internal.models import Team

log = logging.getLogger(__name__)

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

def make_team_repository() -> TeamRepository:
    return TeamRepository()
