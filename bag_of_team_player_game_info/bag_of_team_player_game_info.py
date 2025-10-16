import json
import logging
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
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", 
    datefmt="%Y-%m-%d %H:%M:%S"  # Формат времени
)
log = logging.getLogger(__name__)

# ================= ClickHouse data fetching =================
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
    ) AS t
    USING (game_id)
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
    log.info(f"Получено {len(X)} записей с {X.shape[1]} признаками.")
    return X, y

# ================= Encoders (Sparse) =================
class TeamBagEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = np.asarray(X)
        uniques = np.unique(X.flatten())
        self.d = {val: idx for idx, val in enumerate(uniques)}
        log.info(f"TeamBagEncoder: найдено {len(self.d)} уникальных команд.")
        return self

    def transform(self, X):
        X = np.asarray(X)
        n_samples, n_features = X.shape[0], len(self.d)
        rows, cols, data = [], [], []
        for i, row in enumerate(X):
            for j, val in enumerate(row):
                if val in self.d:
                    rows.append(i)
                    cols.append(self.d[val])
                    data.append(1 if j == 0 else -1)
        log.info(f"TeamBagEncoder: преобразовано {n_samples} строк в разреженный формат.")
        return sparse.csr_matrix((data, (rows, cols)), shape=(n_samples, n_features), dtype=int)


class PlayerBagEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = np.asarray(X)
        uniques = np.unique(X.flatten())
        self.d = {val: idx for idx, val in enumerate(uniques)}
        log.info(f"PlayerBagEncoder: найдено {len(self.d)} уникальных игроков.")
        return self

    def transform(self, X):
        X = np.asarray(X)
        n_samples, n_features = X.shape[0], len(self.d)
        rows, cols, data = [], [], []
        for i, row in enumerate(X):
            for j, val in enumerate(row):
                if val in self.d:
                    rows.append(i)
                    cols.append(self.d[val])
                    data.append(1 if j < 5 else -1)
        log.info(f"PlayerBagEncoder: преобразовано {n_samples} строк в разреженный формат.")
        return sparse.csr_matrix((data, (rows, cols)), shape=(n_samples, n_features), dtype=int)

# ================= L1 Feature Selector =================
class RecursiveFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Рекурсивный отбор признаков с простой LogisticRegression (без регуляризации).
    Permutation importance вычисляется на валидационных фолдах TimeSeriesSplit.
    """
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
            log.info(f"RecursiveFeatureSelector: итерация {iteration}, признаков {len(selected_features)}")
            X_sel = X[:, selected_features]

            tscv = TimeSeriesSplit(n_splits=self.cv_splits)
            perm_importances_fold = []

            for tr_idx, val_idx in tscv.split(X_sel, y):
                # простая логит-модель без регуляризации
                model = LogisticRegression(random_state=self.random_state)
                model.fit(X_sel[tr_idx], y[tr_idx])

                # вычисляем permutation importance на валидационном фолде
                imp = self._permutation_importance(model, X_sel[val_idx], y[val_idx], self.n_repeats)
                perm_importances_fold.append(imp)

            # усредняем permutation importance по фолдам
            avg_perm_importance = {}
            for imp in perm_importances_fold:
                for k, v in imp.items():
                    avg_perm_importance[k] = avg_perm_importance.get(k, 0) + v / len(perm_importances_fold)
            log.info(f"Средние permutation importance: {avg_perm_importance}")

            # оставляем признаки с положительной важностью
            non_zero_idx = np.array([k for k, v in avg_perm_importance.items() if v > 0])
            log.info(f"Выбрано {len(non_zero_idx)} признаков с положительной permutation importance.")

            if len(non_zero_idx) == len(selected_features) or len(non_zero_idx) == 0:
                log.info(f"RecursiveFeatureSelector: отбор завершен. Осталось {len(non_zero_idx)} признаков.")
                break

            selected_features = selected_features[non_zero_idx]

        self.selected_features_ = selected_features
        return self

    def transform(self, X):
        if self.selected_features_ is None:
            raise RuntimeError("RecursiveFeatureSelector еще не был обучен!")
        log.info(f"RecursiveFeatureSelector: трансформация данных с {X.shape[1]} до {len(self.selected_features_)} признаков.")
        return X[:, self.selected_features_]

    def _permutation_importance(self, model, X_val, y_val, n_repeats=5):
        """Permutation importance с multiprocessing для ускорения расчёта."""
        if sparse.issparse(X_val):
            X_val = X_val.tocsr()

        baseline_score = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        n_features = X_val.shape[1]
       
        pool = Pool(processes=cpu_count())
        func = partial(_perm_col_importance, model=model, X_val=X_val, y_val=y_val,
                    baseline_score=baseline_score, n_repeats=n_repeats,
                    rng_seed=self.random_state)
        
        results = list(tqdm(pool.imap(func, range(n_features)), total=n_features))
        pool.close()
        pool.join()

        # собираем результаты в словарь
        importances = {col: score for col, score in results}
        return importances

def _perm_col_importance(col_idx, model, X_val, y_val, baseline_score, n_repeats, rng_seed):
    rng = np.random.default_rng(rng_seed + col_idx)  # разные семена для каждого процесса
    scores = []
    X_val_dense = X_val.toarray() if sparse.issparse(X_val) else X_val
    for _ in range(n_repeats):
        X_perm = X_val_dense.copy()
        rng.shuffle(X_perm[:, col_idx])
        y_pred_perm = model.predict_proba(X_perm)[:, 1]
        scores.append(baseline_score - roc_auc_score(y_val, y_pred_perm))
    return col_idx, np.mean(scores)


def get_metrics(y_test, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    log.info(f"Метрики рассчитаны: accuracy={acc}, precision={prec}, recall={rec}, f1={f1}, auc={auc}")
    return {"accuracy": round(acc,2), "precision": round(prec,2),
            "recall": round(rec,2), "f1": round(f1,2), "auc": round(auc,2),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}

def create_team_player_info_pipeline_with_l1():
    log.info("Создание pipeline с TeamBagEncoder, PlayerBagEncoder и L1FeatureSelector (без GridSearchCV)...")
    TIMESTAMP_IDX = 0
    CATEGORICAL_IDX = [1, 2, 3, 4, 5]
    TEAM_IDX = [6, 7]
    PLAYER_IDX = list(range(8, 18))

    preprocessor = ColumnTransformer(transformers=[
        ("timestamp", MinMaxScaler(), [TIMESTAMP_IDX]),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True), CATEGORICAL_IDX),
        ("teams", TeamBagEncoder(), TEAM_IDX),
        ("players", PlayerBagEncoder(), PLAYER_IDX)
    ])

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("feature_selection", RecursiveFeatureSelector()),
        ("classifier", LogisticRegression(random_state=42, solver="liblinear"))
    ])

    log.info("Pipeline создан и готов к обучению (без GridSearchCV).")
    return pipeline


def bag_of_team_player_game_info(
    split_file="data/train_test_splits/2340df30119543005f8d1baceef714c7.json",
    clickhouse_host="localhost", 
    clickhouse_port=9000,
    clickhouse_user="cs2_user", 
    clickhouse_password="cs2_password",
    clickhouse_db="cs2_db",
):
    log.info("Запуск bag_of_team_player_game_info...")
    client = Client(host=clickhouse_host, port=clickhouse_port,
                    user=clickhouse_user, password=clickhouse_password, database=clickhouse_db)
    log.info(f"Подключение к ClickHouse {clickhouse_host}:{clickhouse_port}, база {clickhouse_db}")

    path = Path(split_file)
    with open(path, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    train_ids = split_data.get("train", [])
    test_ids = split_data.get("test", [])
    log.info(f"Загружены split файлы: train={len(train_ids)}, test={len(test_ids)}")

    X_train, y_train = fetch_games_by_ids(client, train_ids, clickhouse_db)
    X_test, y_test = fetch_games_by_ids(client, test_ids, clickhouse_db)

    pipeline = create_team_player_info_pipeline_with_l1()
    log.info("Начало обучения модели...")
    pipeline.fit(X_train, y_train)
    log.info("Обучение завершено.")

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = get_metrics(y_test, y_proba)
    log.info("bag_of_team_player_game_info завершена.")
    return pipeline, metrics

if __name__ == "__main__":
    pipeline, metrics = bag_of_team_player_game_info()
    print("Метрики классификации (JSON):")
    print(json.dumps(metrics, indent=4))
