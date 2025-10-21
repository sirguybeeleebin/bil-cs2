from app.models import Map, Team, Player
from django.conf import settings
import json
import pickle
from pathlib import Path
import logging
from typing import Generator

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class JsonRepository:
    def __init__(self, path_to_dir: Path):
        self.path_to_dir = path_to_dir
        log.info(f"JsonRepository инициализирован, директория: {self.path_to_dir}")

    def generate(self) -> Generator[dict, None, None]:
        if not self.path_to_dir.exists():
            log.warning(f"Directory does not exist: {self.path_to_dir}")
            return
        for file_path in self.path_to_dir.iterdir():
            if not file_path.is_file():
                continue
            try:
                with file_path.open("r", encoding="utf-8") as f:
                    game = json.load(f)
                log.info(f"Загружена игра из файла {file_path.name}")
                yield game
            except json.JSONDecodeError:
                log.error(f"Не удалось декодировать JSON в файле: {file_path}")
            except Exception as e:
                log.error(f"Ошибка при чтении файла {file_path}: {e}")


class PickleRepository:
    def __init__(self, path_to_dir: Path):
        self.path_to_dir = path_to_dir
        self.path_to_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"PickleRepository инициализирован, директория: {self.path_to_dir}")

    def save(self, obj, filename: str | None = None) -> Path:
        if filename is None:
            obj_bytes = pickle.dumps(obj)
            filename = f"{hash(obj_bytes):x}.pkl"

        file_path = self.path_to_dir / filename
        try:
            with file_path.open("wb") as f:
                pickle.dump(obj, f)
            log.info(f"Объект успешно сохранён в {file_path}")
            return file_path
        except Exception as e:
            log.error(f"Ошибка при сохранении объекта в {file_path}: {e}")
            raise


class MapRepository:
    def upsert(self, map_data: dict) -> Map:
        map_obj, _ = Map.objects.update_or_create(
            map_id=map_data["map_id"],
            defaults={"name": map_data["name"]}
        )
        log.info(f"Map upsert выполнен: {map_data}")
        return map_obj


class TeamRepository:
    def upsert(self, team_data: dict) -> Team:
        team_obj, _ = Team.objects.update_or_create(
            team_id=team_data["team_id"],
            defaults={"name": team_data["name"]}
        )
        log.info(f"Team upsert выполнен: {team_data}")
        return team_obj


class PlayerRepository:
    def upsert(self, player_data: dict) -> Player:
        player_obj, _ = Player.objects.update_or_create(
            player_id=player_data["player_id"],
            defaults={"name": player_data["name"]}
        )
        log.info(f"Player upsert выполнен: {player_data}")
        return player_obj


game_raw_repository = JsonRepository(path_to_dir=settings.PATH_TO_GAMES_RAW_DIR)
ml_results_repository = PickleRepository(path_to_dir=settings.PATH_TO_ML_RESULTS_DIR)

map_repo = MapRepository()
team_repo = TeamRepository()
player_repo = PlayerRepository()
