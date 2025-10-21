import logging
from typing import Optional

from internal.repositories import MapRepository, TeamRepository, PlayerRepository

log = logging.getLogger(__name__)

class CS2Service:
    """Сервис для работы с картами, командами и игроками."""

    def __init__(
        self,
        map_repo: MapRepository,
        team_repo: TeamRepository,
        player_repo: PlayerRepository,
    ):
        self.map_repo = map_repo
        self.team_repo = team_repo
        self.player_repo = player_repo
        log.info("CS2Service инициализирован")

    def get_map_by_name(self, name: str) -> Optional[dict]:
        if not name:
            log.warning("Имя карты не указано")
            return None
        result = self.map_repo.get_by_name(name)
        if result:
            log.info(f"Карта найдена: {result}")
        else:
            log.info(f"Карта '{name}' не найдена")
        return result

    def get_team_by_name(self, name: str) -> Optional[dict]:
        if not name:
            log.warning("Имя команды не указано")
            return None
        result = self.team_repo.get_by_name(name)
        if result:
            log.info(f"Команда найдена: {result}")
        else:
            log.info(f"Команда '{name}' не найдена")
        return result

    def get_player_by_name(self, name: str) -> Optional[dict]:
        if not name:
            log.warning("Имя игрока не указано")
            return None
        result = self.player_repo.get_by_name(name)
        if result:
            log.info(f"Игрок найден: {result}")
        else:
            log.info(f"Игрок '{name}' не найден")
        return result


def make_cs2_service(
    map_repo: MapRepository,
    team_repo: TeamRepository,
    player_repo: PlayerRepository,
) -> CS2Service:
    return CS2Service(map_repo=map_repo, team_repo=team_repo, player_repo=player_repo)
