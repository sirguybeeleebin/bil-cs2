import argparse
import ast
import hashlib
import json
import logging
import pickle
from datetime import datetime, timezone
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pika
from clickhouse_driver import Client
from pydantic_settings import BaseSettings, SettingsConfigDict
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


class Settings(BaseSettings):
    CLICKHOUSE_HOST: str = "localhost"
    CLICKHOUSE_PORT: int = 9000
    CLICKHOUSE_HTTP_PORT: int = 8123
    CLICKHOUSE_USER: str = "cs2_user"
    CLICKHOUSE_PASSWORD: str = "cs2_password"
    CLICKHOUSE_DB: str = "cs2_db"
    RABBITMQ_USER: str = "cs2_user"
    RABBITMQ_PASSWORD: str = "cs2_password"
    RABBITMQ_HOST: str = "localhost"
    RABBITMQ_AMQP_PORT: int = 5672
    RABBITMQ_MANAGEMENT_PORT: int = 15672
    RABBITMQ_EXCHANGE: str = "cs2_exchange"
    RABBITMQ_EXCHANGE_TYPE: str = "direct"
    RABBITMQ_QUEUE: str = "cs2_queue"
    RABBITMQ_ROUTING_KEY_ETL: str = "cs2.etl_completed"
    RABBITMQ_ROUTING_KEY_SPLIT: str = "cs2.split_created"
    RABBITMQ_ROUTING_KEY_ML: str = "cs2.ml_completed"
    OUTPUT_DIR_RAW_SPLITS: str = "data/train_test_splits"
    OUTPUT_DIR_ML: str = "data/ml"
    PATH_TO_GAMES_RAW_DIR: str = "data/games_raw"
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


def fetch_games_by_ids(client: Client, game_ids, clickhouse_db="cs2_db"):
    if not game_ids:
        log.warning("Список game_ids пуст. Возвращаю пустые массивы.")
        return np.array([]), np.array([])
    log.info(f"Получение данных для {len(game_ids)} игр из ClickHouse...")
    ids_str = ",".join(str(i) for i in game_ids)
    query = f"""
    SELECT
        g.game_id,
        any(g.begin_at) AS timestamp,
        any(g.map_id) AS map_id,
        any(g.league_id) AS league_id,
        any(g.serie_id) AS serie_id,
        any(g.tournament_id) AS tournament_id,
        any(g.tier_id) AS tier_id,
        t.teams[1] AS t1_id,
        t.teams[2] AS t2_id,
        arrayElement(arraySlice(arraySort(arrayDistinct(groupArrayIf(player_id, team_id = t.teams[1]))), 1, 5), 1) AS p1_id,
        arrayElement(arraySlice(arraySort(arrayDistinct(groupArrayIf(player_id, team_id = t.teams[1]))), 1, 5), 2) AS p2_id,
        arrayElement(arraySlice(arraySort(arrayDistinct(groupArrayIf(player_id, team_id = t.teams[1]))), 1, 5), 3) AS p3_id,
        arrayElement(arraySlice(arraySort(arrayDistinct(groupArrayIf(player_id, team_id = t.teams[1]))), 1, 5), 4) AS p4_id,
        arrayElement(arraySlice(arraySort(arrayDistinct(groupArrayIf(player_id, team_id = t.teams[1]))), 1, 5), 5) AS p5_id,
        arrayElement(arraySlice(arraySort(arrayDistinct(groupArrayIf(player_id, team_id = t.teams[2]))), 1, 5), 1) AS p6_id,
        arrayElement(arraySlice(arraySort(arrayDistinct(groupArrayIf(player_id, team_id = t.teams[2]))), 1, 5), 2) AS p7_id,
        arrayElement(arraySlice(arraySort(arrayDistinct(groupArrayIf(player_id, team_id = t.teams[2]))), 1, 5), 3) AS p8_id,
        arrayElement(arraySlice(arraySort(arrayDistinct(groupArrayIf(player_id, team_id = t.teams[2]))), 1, 5), 4) AS p9_id,
        arrayElement(arraySlice(arraySort(arrayDistinct(groupArrayIf(player_id, team_id = t.teams[2]))), 1, 5), 5) AS p10_id,
        sumIf(g.round_win, g.team_id = t.teams[1]) > sumIf(g.round_win, g.team_id = t.teams[2]) AS team1_win
    FROM {clickhouse_db}.games_flatten AS g
    INNER JOIN (
        SELECT
            game_id,
            arraySort(arrayDistinct(groupArray(team_id))) AS teams
        FROM {clickhouse_db}.games_flatten
        WHERE game_id IN ({ids_str})
        GROUP BY game_id
    ) AS t USING (game_id)
    GROUP BY g.game_id, t.teams
    ORDER BY timestamp ASC
    """
    result = client.execute(query, with_column_types=True)
    if not result:
        log.warning("Запрос вернул пустой результат.")
        return np.array([]), np.array([])
    rows = [list(r) for r in result[0]]
    for row in rows:
        if hasattr(row[1], "timestamp"):
            row[1] = row[1].timestamp()
    X = np.array([r[1:-1] for r in rows], dtype=float)
    y = np.array([r[-1] for r in rows], dtype=int)
    log.info(f"Получено {len(X)} строк с {X.shape[1]} признаками.")
    return X, y


class TeamBagEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = np.asarray(X)
        self.d = {val: idx for idx, val in enumerate(np.unique(X.flatten()))}
        log.info(f"TeamBagEncoder: найдено {len(self.d)} уникальных команд.")
        return self

    def transform(self, X):
        X = np.asarray(X)
        rows, cols, data = [], [], []
        for i, row in enumerate(X):
            for j, val in enumerate(row):
                if val in self.d:
                    rows.append(i)
                    cols.append(self.d[val])
                    data.append(1 if j == 0 else -1)
        return sparse.csr_matrix(
            (data, (rows, cols)), shape=(X.shape[0], len(self.d)), dtype=int
        )


class PlayerBagEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = np.asarray(X)
        self.d = {val: idx for idx, val in enumerate(np.unique(X.flatten()))}
        log.info(f"PlayerBagEncoder: найдено {len(self.d)} уникальных игроков.")
        return self

    def transform(self, X):
        X = np.asarray(X)
        rows, cols, data = [], [], []
        for i, row in enumerate(X):
            for j, val in enumerate(row):
                if val in self.d:
                    rows.append(i)
                    cols.append(self.d[val])
                    data.append(1 if j < 5 else -1)
        return sparse.csr_matrix(
            (data, (rows, cols)), shape=(X.shape[0], len(self.d)), dtype=int
        )


class RecursiveFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, random_state=42, cv_splits=10, n_repeats=1):
        self.random_state = random_state
        self.cv_splits = cv_splits
        self.n_repeats = n_repeats
        self.selected_features_ = None

    def fit(self, X, y=None):
        X = X if sparse.issparse(X) else np.asarray(X)
        y = np.asarray(y)
        selected_features = np.arange(X.shape[1])
        iteration = 0
        while True:
            iteration += 1
            log.info(f"RFS итерация {iteration}, признаков={len(selected_features)}")
            X_sel = X[:, selected_features]
            tscv = TimeSeriesSplit(n_splits=self.cv_splits)
            fold_importances = []
            for tr_idx, val_idx in tscv.split(X_sel, y):
                model = LogisticRegression(random_state=self.random_state)
                model.fit(X_sel[tr_idx], y[tr_idx])
                imp = self._permutation_importance(
                    model, X_sel[val_idx], y[val_idx], self.n_repeats
                )
                fold_importances.append(imp)
            avg_importance = {}
            for imp in fold_importances:
                for k, v in imp.items():
                    avg_importance[k] = avg_importance.get(k, 0) + v / len(
                        fold_importances
                    )
            non_zero_idx = np.array([k for k, v in avg_importance.items() if v > 0])
            if len(non_zero_idx) == len(selected_features) or len(non_zero_idx) == 0:
                break
            selected_features = selected_features[non_zero_idx]
        self.selected_features_ = selected_features
        return self

    def transform(self, X):
        if self.selected_features_ is None:
            raise RuntimeError("RFS не обучен!")
        return X[:, self.selected_features_]

    def _permutation_importance(self, model, X_val, y_val, n_repeats=5):
        if sparse.issparse(X_val):
            X_val = X_val.tocsr()
        baseline = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        pool = Pool(cpu_count())
        func = partial(
            perm_col_importance,
            model=model,
            X_val=X_val,
            y_val=y_val,
            baseline_score=baseline,
            n_repeats=n_repeats,
            rng_seed=self.random_state,
        )
        results = list(pool.imap(func, range(X_val.shape[1])))
        pool.close()
        pool.join()
        return {col: score for col, score in results}


def perm_col_importance(
    col_idx, model, X_val, y_val, baseline_score, n_repeats, rng_seed
):
    rng = np.random.default_rng(rng_seed + col_idx)
    scores = []
    X_dense = X_val.toarray() if sparse.issparse(X_val) else X_val
    for _ in range(n_repeats):
        X_perm = X_dense.copy()
        rng.shuffle(X_perm[:, col_idx])
        scores.append(
            baseline_score - roc_auc_score(y_val, model.predict_proba(X_perm)[:, 1])
        )
    return col_idx, np.mean(scores)


def create_pipeline():
    TIMESTAMP_IDX = 0
    CATEGORICAL_IDX = [1, 2, 3, 4, 5]
    TEAM_IDX = [6, 7]
    PLAYER_IDX = list(range(8, 18))
    preprocessor = ColumnTransformer(
        [
            ("timestamp", MinMaxScaler(), [TIMESTAMP_IDX]),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                CATEGORICAL_IDX,
            ),
            ("teams", TeamBagEncoder(), TEAM_IDX),
            ("players", PlayerBagEncoder(), PLAYER_IDX),
        ]
    )
    return Pipeline(
        [
            ("preprocessing", preprocessor),
            ("feature_selection", RecursiveFeatureSelector()),
            ("classifier", LogisticRegression(random_state=42, solver="liblinear")),
        ]
    )


def get_metrics(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": round(acc, 2),
        "precision": round(prec, 2),
        "recall": round(rec, 2),
        "f1": round(f1, 2),
        "auc": round(auc, 2),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def run_pipeline(client: Client, train_ids, test_ids, clickhouse_db):
    X_train, y_train = fetch_games_by_ids(client, train_ids, clickhouse_db)
    X_test, y_test = fetch_games_by_ids(client, test_ids, clickhouse_db)
    if len(X_train) == 0 or len(X_test) == 0:
        log.warning("Пустые данные train/test, пропуск выполнения pipeline.")
        return None, None
    pipeline = create_pipeline()
    log.info("Обучение pipeline...")
    pipeline.fit(X_train, y_train)
    log.info("Pipeline обучен.")
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = get_metrics(y_test, y_proba)
    log.info(f"Метрики pipeline: {metrics}")
    return pipeline, metrics


def get_settings() -> Settings:
    parser = argparse.ArgumentParser(description="ML consumer for split messages")
    parser.add_argument("--env-file", type=str, default=".env")
    args = parser.parse_args()
    env_path = Path(args.env_file)
    if env_path.exists():
        log.info(f"Загрузка конфигурации из {env_path}")
        return Settings(_env_file=env_path)
    log.warning(f"Файл {env_path} не найден, используются настройки по умолчанию")
    return Settings()


def handle_message(body: bytes, channel, method, settings: Settings, client: Client):
    try:
        log.info(f"Получено сообщение с разбиением: {body.decode()}")
        data = ast.literal_eval(body.decode())
        train_ids = data.get("train", [])
        test_ids = data.get("test", [])
        if not train_ids and not test_ids:
            log.warning("Пустое разбиение, пропуск обучения.")
            return
        log.info(
            f"Запуск ML pipeline для {len(train_ids)} train и {len(test_ids)} test образцов."
        )
        pipeline, metrics = run_pipeline(
            client, train_ids, test_ids, settings.CLICKHOUSE_DB
        )
        if pipeline is None:
            log.warning("Pipeline пропущен из-за пустых данных.")
            return
        hash_input = ",".join(str(x) for x in train_ids + test_ids)
        hash_id = hashlib.md5(hash_input.encode("utf-8")).hexdigest()
        result = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hash_id": hash_id,
            "metrics": metrics,
        }
        output_dir = Path(settings.OUTPUT_DIR_ML)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_file = output_dir / f"{hash_id}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        model_file = output_dir / f"{hash_id}.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(pipeline, f)
        log.info(
            f"Результаты ML сохранены: метрики — {json_file}, модель — {model_file}"
        )
        channel.basic_publish(
            exchange=settings.RABBITMQ_EXCHANGE,
            routing_key=settings.RABBITMQ_ROUTING_KEY_ML,
            body=json.dumps(result),
        )
        log.info(
            f"Сообщение о завершении ML отправлено: {settings.RABBITMQ_ROUTING_KEY_ML}"
        )
    except Exception as e:
        log.exception(f"Ошибка при обработке сообщения: {e}")
    finally:
        if method:
            channel.basic_ack(delivery_tag=method.delivery_tag)


def main():
    settings = get_settings()
    credentials = pika.PlainCredentials(
        settings.RABBITMQ_USER, settings.RABBITMQ_PASSWORD
    )
    parameters = pika.ConnectionParameters(
        host=settings.RABBITMQ_HOST,
        port=settings.RABBITMQ_AMQP_PORT,
        credentials=credentials,
    )
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()
    channel.exchange_declare(
        exchange=settings.RABBITMQ_EXCHANGE,
        exchange_type=settings.RABBITMQ_EXCHANGE_TYPE,
        durable=True,
    )
    channel.queue_declare(queue=settings.RABBITMQ_QUEUE, durable=True)
    channel.queue_bind(
        queue=settings.RABBITMQ_QUEUE,
        exchange=settings.RABBITMQ_EXCHANGE,
        routing_key=settings.RABBITMQ_ROUTING_KEY_SPLIT,
    )
    client = Client(
        host=settings.CLICKHOUSE_HOST,
        port=settings.CLICKHOUSE_PORT,
        user=settings.CLICKHOUSE_USER,
        password=settings.CLICKHOUSE_PASSWORD,
        database=settings.CLICKHOUSE_DB,
    )
    log.info("Ожидание сообщений с разбиениями для запуска ML pipeline...")

    def callback(ch, method, properties, body):
        handle_message(body, ch, method, settings, client)

    channel.basic_consume(queue=settings.RABBITMQ_QUEUE, on_message_callback=callback)
    channel.start_consuming()


if __name__ == "__main__":
    main()
