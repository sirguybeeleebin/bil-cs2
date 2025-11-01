import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable

from celery.result import AsyncResult

from backend.repositories import MapRepository, PlayerRepository, TeamRepository

log = logging.getLogger(__name__)


class DictionaryService:
    def __init__(
        self,
        map_repository: MapRepository,
        team_repository: TeamRepository,
        player_repository: PlayerRepository,
    ):
        self.map_repository = map_repository
        self.team_repository = team_repository
        self.player_repository = player_repository
        log.info("DictionaryService инициализирован")

    def load_dictionaries_from_json(self, path_to_games_raw_dir: Path):
        log.info(f"Загрузка словарей из {path_to_games_raw_dir}")
        for file_path in path_to_games_raw_dir.glob("*.json"):
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    game = json.load(f)
                except json.JSONDecodeError:
                    log.warning(f"Ошибка чтения JSON: {file_path}")
                    continue
                if not self._validate_game(game):
                    log.warning(f"Файл не прошёл валидацию: {file_path}")
                    continue
                self.map_repository.save(
                    map_id=game["map"]["id"], name=game["map"]["name"]
                )
                for p in game["players"]:
                    self.team_repository.save(
                        team_id=p["team"]["id"], name=p["team"]["name"]
                    )
                    self.player_repository.save(
                        player_id=p["player"]["id"], name=p["player"]["name"]
                    )

    def _validate_game(self, game: dict) -> bool:
        try:
            int(game["map"]["id"])
            datetime.fromisoformat(game["begin_at"].replace("Z", "+00:00"))
            team_players: dict = defaultdict(list)
            for p in game["players"]:
                team_players[p["team"]["id"]].append(p["player"]["id"])
            if len(team_players) != 2:
                return False
            for _, p_ids in team_players.items():
                if len(set(p_ids)) != 5:
                    return False
            t1_id, t2_id = list(team_players.keys())
            rounds: list[int] = []
            for r in game["rounds"]:
                if r["round"] is None:
                    continue
                if r["winner_team"] not in (t1_id, t2_id):
                    return False
                rounds.append(r["round"])
            if not rounds or min(rounds) != 1 or max(rounds) < 16:
                return False
            return True
        except Exception:
            return False

    def search_map_by_name(
        self, name: str, page: int = 1, per_page: int = 10
    ) -> list[dict]:
        log.info(f"Поиск карт по имени: {name}, страница {page}")
        offset = (page - 1) * per_page
        return self.map_repository.search_by_name(
            name=name, limit=per_page, offset=offset
        )

    def search_team_by_name(
        self, name: str, page: int = 1, per_page: int = 10
    ) -> list[dict]:
        log.info(f"Поиск команд по имени: {name}, страница {page}")
        offset = (page - 1) * per_page
        return self.team_repository.search_by_name(
            name=name, limit=per_page, offset=offset
        )

    def search_player_by_name(
        self, name: str, page: int = 1, per_page: int = 10
    ) -> list[dict]:
        log.info(f"Поиск игроков по имени: {name}, страница {page}")
        offset = (page - 1) * per_page
        return self.player_repository.search_by_name(
            name=name, limit=per_page, offset=offset
        )


class ForecasterService:
    def __init__(self, inference_model_task: Callable):
        self.inference_model_task = inference_model_task
        log.info("ForecasterService инициализирован")

    def forecast(
        self,
        map_id: int,
        team1_id: int,
        team2_id: int,
        team1_player1_id: int,
        team1_player2_id: int,
        team1_player3_id: int,
        team1_player4_id: int,
        team1_player5_id: int,
        team2_player1_id: int,
        team2_player2_id: int,
        team2_player3_id: int,
        team2_player4_id: int,
        team2_player5_id: int,
    ) -> str:
        log.info(
            f"Формирование прогноза для карты {map_id} и команд {team1_id} vs {team2_id}"
        )
        team_ids = [team1_id, team2_id]
        player_ids = [
            sorted(
                [
                    team1_player1_id,
                    team1_player2_id,
                    team1_player3_id,
                    team1_player4_id,
                    team1_player5_id,
                ]
            ),
            sorted(
                [
                    team2_player1_id,
                    team2_player2_id,
                    team2_player3_id,
                    team2_player4_id,
                    team2_player5_id,
                ]
            ),
        ]
        team_ids_sorted = sorted(team_ids)
        if team_ids != team_ids_sorted:
            team_ids = team_ids_sorted
            player_ids = [player_ids[1], player_ids[0]]
        X = [
            [
                map_id,
                team_ids[0],
                team_ids[1],
                player_ids[0][0],
                player_ids[0][1],
                player_ids[0][2],
                player_ids[0][3],
                player_ids[0][4],
                player_ids[1][0],
                player_ids[1][1],
                player_ids[1][2],
                player_ids[1][3],
                player_ids[1][4],
            ]
        ]
        task = self.inference_model_task.delay(X)
        log.info(f"Прогноз запущен, ID задачи: {task.id}")
        return task.id

    def get_forecast_result_by_id(self, forecast_id: str) -> dict:
        log.info(f"Запрос результата прогноза по ID: {forecast_id}")
        async_result = AsyncResult(forecast_id)

        result_dict = {
            "status": async_result.state,
            "forecast_id": forecast_id,
        }
        if async_result.state == "SUCCESS":
            result_dict["result"] = async_result.result
        return result_dict
