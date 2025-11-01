import logging
from uuid import UUID

from backend.models import Map, Player, Team, TrainMetric, TrainResult, TrainTestSplit

log = logging.getLogger(__name__)


class MapRepository:
    def save(self, map_id: int, name: str) -> dict | None:
        try:
            obj, _ = Map.objects.update_or_create(
                map_id=map_id,
                defaults={"name": name},
            )
            log.info(f"Карта сохранена: {obj.map_id} - {obj.name}")
            return obj.__dict__
        except Exception as e:
            log.error(f"Ошибка при сохранении карты {map_id}: {e}")
            return None

    def search_by_name(self, name: str, limit: int, offset: int) -> list[dict]:
        try:
            qs = Map.objects.filter(name__icontains=name).order_by("map_id")[
                offset : offset + limit
            ]
            log.info(f"Поиск карт по имени '{name}' выполнен, найдено: {qs.count()}")
            return [m.__dict__ for m in qs]
        except Exception as e:
            log.error(f"Ошибка при поиске карт по имени '{name}': {e}")
            return []


class TeamRepository:
    def save(self, team_id: int, name: str) -> dict | None:
        try:
            obj, _ = Team.objects.update_or_create(
                team_id=team_id,
                defaults={"name": name},
            )
            log.info(f"Команда сохранена: {obj.team_id} - {obj.name}")
            return obj.__dict__
        except Exception as e:
            log.error(f"Ошибка при сохранении команды {team_id}: {e}")
            return None

    def search_by_name(self, name: str, limit: int, offset: int) -> list[dict]:
        try:
            qs = Team.objects.filter(name__icontains=name).order_by("team_id")[
                offset : offset + limit
            ]
            log.info(f"Поиск команд по имени '{name}' выполнен, найдено: {qs.count()}")
            return [t.__dict__ for t in qs]
        except Exception as e:
            log.error(f"Ошибка при поиске команд по имени '{name}': {e}")
            return []


class PlayerRepository:
    def save(self, player_id: int, name: str) -> dict | None:
        try:
            obj, _ = Player.objects.update_or_create(
                player_id=player_id,
                defaults={"name": name},
            )
            log.info(f"Игрок сохранён: {obj.player_id} - {obj.name}")
            return obj.__dict__
        except Exception as e:
            log.error(f"Ошибка при сохранении игрока {player_id}: {e}")
            return None

    def search_by_name(self, name: str, limit: int, offset: int) -> list[dict]:
        try:
            qs = Player.objects.filter(name__icontains=name).order_by("player_id")[
                offset : offset + limit
            ]
            log.info(f"Поиск игроков по имени '{name}' выполнен, найдено: {qs.count()}")
            return [p.__dict__ for p in qs]
        except Exception as e:
            log.error(f"Ошибка при поиске игроков по имени '{name}': {e}")
            return []


class TrainTestSplitRepository:
    def save(
        self,
        train_test_split_hash: str,
        game_ids_train: list[int],
        game_ids_test: list[int],
        begin_at_min: str | None = None,
        begin_at_max: str | None = None,
    ) -> dict | None:
        try:
            defaults = {
                "game_ids_train": game_ids_train,
                "game_ids_test": game_ids_test,
                "begin_at_min": begin_at_min,
                "begin_at_max": begin_at_max,
            }

            obj, created = TrainTestSplit.objects.update_or_create(
                train_test_split_hash=train_test_split_hash,
                defaults=defaults,
            )
            log.info(
                f"TrainTestSplit сохранён: {obj.train_test_split_hash} (новый: {created})"
            )
            return obj.__dict__
        except Exception as e:
            log.error(
                f"Ошибка при сохранении TrainTestSplit {train_test_split_hash}: {e}"
            )
            return None


class TrainResultRepository:
    def save(
        self, train_result_id: UUID, train_test_split_hash: str, path_to_model: str
    ) -> dict | None:
        try:
            split_obj = TrainTestSplit.objects.get(
                train_test_split_hash=train_test_split_hash
            )
            obj = TrainResult.objects.create(
                train_result_id=train_result_id,
                train_test_split=split_obj,
                path_to_model=path_to_model,
            )
            log.info(f"TrainResult сохранён: {obj.train_result_id}")
            return obj.__dict__
        except Exception as e:
            log.error(
                f"Ошибка при сохранении TrainResult для split {train_test_split_hash}: {e}"
            )
            return None

    def get_last(self) -> dict | None:
        try:
            obj = TrainResult.objects.order_by("-created_at").first()
            if obj:
                log.info(f"Последний TrainResult получен: {obj.train_result_id}")
                return obj.__dict__
            else:
                log.info("TrainResult не найдено")
                return None
        except Exception as e:
            log.error(f"Ошибка при получении последнего TrainResult: {e}")
            return None


class TrainMetricRepository:
    def save(
        self,
        train_metric_id: UUID,
        train_result_id: UUID,
        auc: float = None,
        f1: float = None,
        precision: float = None,
        recall: float = None,
        accuracy: float = None,
        tp: int = None,
        tn: int = None,
        fp: int = None,
        fn: int = None,
    ) -> dict | None:
        try:
            train_result = TrainResult.objects.get(train_result_id=train_result_id)
            obj = TrainMetric.objects.create(
                train_metric_id=train_metric_id,
                train_result=train_result,
                auc=auc,
                f1=f1,
                precision=precision,
                recall=recall,
                accuracy=accuracy,
                tp=tp,
                tn=tn,
                fp=fp,
                fn=fn,
            )
            log.info(f"TrainMetric сохранён: {obj.train_metric_id}")
            return obj.__dict__
        except Exception as e:
            log.error(
                f"Ошибка при сохранении TrainMetric для train_result {train_result_id}: {e}"
            )
            return None
