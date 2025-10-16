import json
import logging
import pickle
from pathlib import Path
from clickhouse_driver import Client
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from multiprocessing import Pool, cpu_count
from functools import partial
from pydantic_settings import BaseSettings, SettingsConfigDict
import argparse
import pika

# ================= Logging =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# ================= Settings =================
class Settings(BaseSettings):
    clickhouse_host: str = "localhost"
    clickhouse_port: int = 9000
    clickhouse_user: str = "cs2_user"
    clickhouse_password: str = "cs2_password"
    clickhouse_db: str = "cs2_db"

    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/%2F"
    rabbitmq_exchange: str = "cs2"
    rabbitmq_exchange_type: str = "direct"

    rabbitmq_consume_queue: str = "cs2_split_created_queue"
    rabbitmq_consume_routing_key: str = "cs2.split_created"

    rabbitmq_publish_queue: str = "cs2_ml_completed_queue"
    rabbitmq_publish_routing_key: str = "cs2.ml_completed"

    output_dir: str = "data/ml"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

# ================= Fetch data =================
def _fetch_games_by_ids(client: Client, game_ids, clickhouse_db="cs2_db"):
    if not game_ids:
        log.warning("Список game_ids пуст. Возвращаю пустые массивы.")
        return np.array([]), np.array([])

    log.info(f"Fetching data for {len(game_ids)} games from ClickHouse...")
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
        log.warning("Query returned empty result.")
        return np.array([]), np.array([])

    rows = [list(r) for r in result[0]]
    for row in rows:
        if hasattr(row[1], "timestamp"):
            row[1] = row[1].timestamp()
    X = np.array([r[1:-1] for r in rows], dtype=float)
    y = np.array([r[-1] for r in rows], dtype=int)
    log.info(f"Fetched {len(X)} rows with {X.shape[1]} features.")
    return X, y

# ================= Encoders =================
class TeamBagEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = np.asarray(X)
        self.d = {val: idx for idx, val in enumerate(np.unique(X.flatten()))}
        log.info(f"TeamBagEncoder: {len(self.d)} unique teams found.")
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
        return sparse.csr_matrix((data, (rows, cols)), shape=(X.shape[0], len(self.d)), dtype=int)

class PlayerBagEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = np.asarray(X)
        self.d = {val: idx for idx, val in enumerate(np.unique(X.flatten()))}
        log.info(f"PlayerBagEncoder: {len(self.d)} unique players found.")
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
        return sparse.csr_matrix((data, (rows, cols)), shape=(X.shape[0], len(self.d)), dtype=int)

# ================= Recursive Feature Selector =================
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
            log.info(f"RFS iteration {iteration}, features={len(selected_features)}")
            X_sel = X[:, selected_features]
            tscv = TimeSeriesSplit(n_splits=self.cv_splits)
            fold_importances = []
            for tr_idx, val_idx in tscv.split(X_sel, y):
                model = LogisticRegression(random_state=self.random_state)
                model.fit(X_sel[tr_idx], y[tr_idx])
                imp = self._permutation_importance(model, X_sel[val_idx], y[val_idx], self.n_repeats)
                fold_importances.append(imp)
            avg_importance = {}
            for imp in fold_importances:
                for k, v in imp.items():
                    avg_importance[k] = avg_importance.get(k, 0) + v / len(fold_importances)
            non_zero_idx = np.array([k for k, v in avg_importance.items() if v > 0])
            if len(non_zero_idx) == len(selected_features) or len(non_zero_idx) == 0:
                break
            selected_features = selected_features[non_zero_idx]
        self.selected_features_ = selected_features
        return self

    def transform(self, X):
        if self.selected_features_ is None:
            raise RuntimeError("RFS not fitted yet!")
        return X[:, self.selected_features_]

    def _permutation_importance(self, model, X_val, y_val, n_repeats=5):
        if sparse.issparse(X_val):
            X_val = X_val.tocsr()
        baseline = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        pool = Pool(cpu_count())
        func = partial(_perm_col_importance, model=model, X_val=X_val, y_val=y_val,
                       baseline_score=baseline, n_repeats=n_repeats, rng_seed=self.random_state)
        results = list(pool.imap(func, range(X_val.shape[1])))
        pool.close(); pool.join()
        return {col: score for col, score in results}

def _perm_col_importance(col_idx, model, X_val, y_val, baseline_score, n_repeats, rng_seed):
    rng = np.random.default_rng(rng_seed + col_idx)
    scores = []
    X_dense = X_val.toarray() if sparse.issparse(X_val) else X_val
    for _ in range(n_repeats):
        X_perm = X_dense.copy()
        rng.shuffle(X_perm[:, col_idx])
        scores.append(baseline_score - roc_auc_score(y_val, model.predict_proba(X_perm)[:, 1]))
    return col_idx, np.mean(scores)

# ================= Pipeline =================
def _create_pipeline():
    TIMESTAMP_IDX = 0
    CATEGORICAL_IDX = [1, 2, 3, 4, 5]
    TEAM_IDX = [6, 7]
    PLAYER_IDX = list(range(8, 18))
    preprocessor = ColumnTransformer([
        ("timestamp", MinMaxScaler(), [TIMESTAMP_IDX]),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True), CATEGORICAL_IDX),
        ("teams", TeamBagEncoder(), TEAM_IDX),
        ("players", PlayerBagEncoder(), PLAYER_IDX)
    ])
    return Pipeline([
        ("preprocessing", preprocessor),
        ("feature_selection", RecursiveFeatureSelector()),
        ("classifier", LogisticRegression(random_state=42, solver="liblinear"))
    ])

# ================= Metrics =================
def _get_metrics(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {"accuracy": round(acc,2), "precision": round(prec,2),
            "recall": round(rec,2), "f1": round(f1,2), "auc": round(auc,2),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}

# ================= Pipeline Runner =================
def _run_pipeline(client: Client, train_ids, test_ids, clickhouse_db):
    X_train, y_train = _fetch_games_by_ids(client, train_ids, clickhouse_db)
    X_test, y_test = _fetch_games_by_ids(client, test_ids, clickhouse_db)

    pipeline = _create_pipeline()
    log.info("Training pipeline...")
    pipeline.fit(X_train, y_train)
    log.info("Pipeline trained.")

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = _get_metrics(y_test, y_proba)
    log.info(f"Pipeline metrics: {metrics}")
    return pipeline, metrics

# ================= RabbitMQ =================
def consume_rabbit(client: Client, clickhouse_db: str, channel, consume_queue: str, exchange: str, publish_queue: str, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    def callback(ch, method, properties, body):
        message = json.loads(body)
        split_file = message.get("split_file")
        hash_id = Path(split_file).stem
        log.info(f"Received split_file: {split_file} (hash_id={hash_id})")

        with open(split_file, "r", encoding="utf-8") as f:
            split_data = json.load(f)
        train_ids, test_ids = split_data.get("train", []), split_data.get("test", [])

        pipeline, metrics = _run_pipeline(client, train_ids, test_ids, clickhouse_db)

        # Save pipeline
        pipeline_path = Path(output_dir) / f"{hash_id}.pkl"
        with open(pipeline_path, "wb") as f:
            pickle.dump(pipeline, f)
        log.info(f"Saved pipeline to {pipeline_path}")

        # Save metrics
        metrics_path = Path(output_dir) / f"{hash_id}.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        log.info(f"Saved metrics to {metrics_path}")

        # Publish to RabbitMQ
        _publish_rabbit(channel, publish_queue, exchange, metrics)
        ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_consume(queue=consume_queue, on_message_callback=callback)
    channel.start_consuming()

def _publish_rabbit(channel, publish_queue: str, exchange: str, message: dict):
    channel.basic_publish(
        exchange=exchange,
        routing_key=publish_queue,
        body=json.dumps(message),
        properties=pika.BasicProperties(content_type='application/json', delivery_mode=2)
    )
    log.info(f"Published metrics: {message}")

def _get_settings() -> Settings:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=str, default=".env")
    args = parser.parse_args()
    env_path = Path(args.env_file)
    return Settings(_env_file=env_path) if env_path.exists() else Settings()

def main():
    settings = _get_settings()
    client = Client(
        host=settings.clickhouse_host,
        port=settings.clickhouse_port,
        user=settings.clickhouse_user,
        password=settings.clickhouse_password,
        database=settings.clickhouse_db
    )

    connection = pika.BlockingConnection(pika.URLParameters(settings.rabbitmq_url))
    channel = connection.channel()

    # Declare exchange
    channel.exchange_declare(
        exchange=settings.rabbitmq_exchange,
        exchange_type=settings.rabbitmq_exchange_type,
        durable=True
    )

    # Declare consume queue and bind
    channel.queue_declare(queue=settings.rabbitmq_consume_queue, durable=True)
    channel.queue_bind(
        queue=settings.rabbitmq_consume_queue,
        exchange=settings.rabbitmq_exchange,
        routing_key=settings.rabbitmq_consume_routing_key
    )
    log.info(f"Waiting for messages on exchange '{settings.rabbitmq_exchange}' with routing key '{settings.rabbitmq_consume_routing_key}'...")

    # Start consuming messages
    consume_rabbit(
        client=client,
        clickhouse_db=settings.clickhouse_db,
        channel=channel,
        consume_queue=settings.rabbitmq_consume_queue,
        exchange=settings.rabbitmq_exchange,
        publish_queue=settings.rabbitmq_publish_queue,
        output_dir=settings.output_dir
    )

if __name__ == "__main__":
    main()
