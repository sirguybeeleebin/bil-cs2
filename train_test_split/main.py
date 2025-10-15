import os
import json
from collections import defaultdict, Counter
from dateutil.parser import parse
import numpy as np
import hashlib
import argparse
import logging

# -----------------------------
# Настройка логирования
# -----------------------------
log = logging.getLogger("TrainTestSplit")
log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
log.addHandler(ch)


def hash_ids(game_ids_list: list) -> str:
    """Возвращает SHA256 хэш объединённых ID игр."""
    concat_ids = ",".join(map(str, game_ids_list))
    return hashlib.sha256(concat_ids.encode("utf-8")).hexdigest()[:16]


def train_test_split_games(path_to_raw_dir: str, path_to_split_dir: str, test_size: int):
    begins_at = []
    game_ids = []

    os.makedirs(path_to_split_dir, exist_ok=True)
    log.info("Начало обработки игр из %s", path_to_raw_dir)

    for game_file in os.listdir(path_to_raw_dir):
        try:
            with open(os.path.join(path_to_raw_dir, game_file), "r", encoding="utf-8") as f:
                game_data = json.load(f)

            # Проверка даты начала игры
            game_begin = parse(game_data.get("begin_at", ""))
            if not game_begin:
                log.warning("Пропуск файла %s: отсутствует begin_at", game_file)
                continue

            # Сбор игроков по командам
            teams_players = defaultdict(list)
            for p in game_data.get("players", []):
                teams_players[p["team"]["id"]].append(p["player"]["id"])

            if len(teams_players) != 2 or any(len(players) != 5 for players in teams_players.values()):
                log.warning("Пропуск файла %s: некорректное количество команд или игроков", game_file)
                continue

            t1_id, t2_id = sorted(teams_players.keys())

            # Подсчёт победителей раундов
            rounds = game_data.get("rounds", [])
            winner_counts = Counter()
            for rnd in rounds:
                winner = rnd.get("winner_team")
                if winner in (t1_id, t2_id):
                    winner_counts[winner] += 1

            if not winner_counts:
                log.warning("Пропуск файла %s: отсутствуют корректные победители раундов", game_file)
                continue

            winner_team = winner_counts.most_common(1)[0][0]
            if winner_team not in (t1_id, t2_id):
                log.warning("Пропуск файла %s: победитель не соответствует командам", game_file)
                continue

            game_ids.append(game_data["id"])
            begins_at.append(game_begin)

        except Exception as e:
            log.error("Ошибка при обработке файла %s: %s", game_file, e)
            continue

    if not game_ids:
        log.error("Не найдено ни одной корректной игры. Завершение работы.")
        return

    log.info("Всего обработано корректных игр: %d", len(game_ids))

    # Сортировка по дате начала игры
    order = np.argsort(begins_at)
    game_ids_sorted = np.array(game_ids)[order]

    # Разделение на обучающую и тестовую выборки
    games_train = game_ids_sorted[:-test_size].tolist()
    games_test = game_ids_sorted[-test_size:].tolist()
    log.info("Количество игр для обучения: %d, для теста: %d", len(games_train), len(games_test))

    # Генерация хэша для имени файла
    hash_id = hash_ids(game_ids_sorted.tolist())
    output_file = os.path.join(path_to_split_dir, f"{hash_id}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"train": games_train, "test": games_test}, f, indent=4)

    log.info("Файл разбиения train/test сохранён: %s", output_file)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Разделение JSON игр на обучающую и тестовую выборки с сохранением файла по хэшу."
    )
    parser.add_argument(
        "--path_to_games_raw_dir",
        type=str,
        default="data/games_raw",
        help="Путь к директории с исходными JSON файлами игр."
    )
    parser.add_argument(
        "--path_to_train_test_splits_dir",
        type=str,
        default="data/train_test_splits",
        help="Директория для сохранения JSON файла с train/test разбиением."
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=100,
        help="Количество игр в тестовой выборке."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_test_split_games(
        args.path_to_games_raw_dir,
        args.path_to_train_test_splits_dir,
        args.test_size
    )


if __name__ == "__main__":
    main()
