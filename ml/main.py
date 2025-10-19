import os
import json
from dateutil.parser import parse
from collections import defaultdict
import numpy as np
from tqdm.notebook import tqdm
from sklearn.base import BaseEstimator, TransformerMixin, clone
from scipy import sparse
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import hashlib
import pickle

# ----------------------- Data loading -----------------------
def generate_game_raw(path_to_games_raw_dir: str = "data/games_raw"):
    for filename in tqdm(os.listdir(path_to_games_raw_dir)):
        file_path = os.path.join(path_to_games_raw_dir, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                yield json.load(f)
        except:
            continue

def validate_game(game: dict) -> bool:
    try:
        int(game["id"])
        parse(game["begin_at"])
        int(game["match"]["league"]["id"])
        int(game["match"]["serie"]["id"])
        int(game["match"]["tournament"]["id"])
        int(game["map"]["id"])
        team_players = defaultdict(list)
        for p in game["players"]:
            team_players[p["team"]["id"]].append(p["player"]["id"])
        if len(team_players) != 2:
            return False
        for p_ids in team_players.values():
            if len(set(p_ids)) != 5:
                return False
        team_ids = list(team_players.keys())
        rounds = []
        for r in game["rounds"]:
            if r["round"] is None or r["ct"] not in team_ids or r["terrorists"] not in team_ids or r["winner_team"] not in team_ids:
                return False
            rounds.append(r["round"])
        if min(rounds) != 1 or max(rounds) < 16:
            return False
        return True
    except:
        return False

def get_game_ids(path_to_games_raw_dir: str = "data/games_raw") -> list[int]:
    game_ids_valid, game_begin_at_valid = [], []
    for game in generate_game_raw(path_to_games_raw_dir):
        if validate_game(game):
            game_ids_valid.append(game["id"])
            game_begin_at_valid.append(parse(game["begin_at"]))
    return np.array(game_ids_valid)[np.argsort(game_begin_at_valid)].tolist()

def get_X_y(path_to_games_raw: str, game_ids: list[int]):
    X, y = [], []
    for game_id in tqdm(game_ids):
        file_path = os.path.join(path_to_games_raw, f"{game_id}.json")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                game = json.load(f)
            team_players = defaultdict(list)
            for p in game["players"]:
                team_players[p["team"]["id"]].append(p["player"]["id"])
            t1_id, t2_id = sorted(team_players.keys())
            X.append([t1_id, t2_id] + sorted(team_players[t1_id]) + sorted(team_players[t2_id]))
            team_win_count = {t1_id: 0, t2_id: 0}
            for r in game["rounds"]:
                team_win_count[r["winner_team"]] += 1
            y.append(int(team_win_count[t1_id] > team_win_count[t2_id]))
        except:
            continue
    return np.array(X), np.array(y)

# ----------------------- Column selector -----------------------
class ColumnSelector(BaseEstimator, TransformerMixin):   
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)
        return X[:, self.columns]

# ----------------------- Bag encoders -----------------------
class PlayerBagEncoder(BaseEstimator, TransformerMixin):   
    def __init__(self):
        self.player_dict = None

    def fit(self, X, y=None):      
        X = np.asarray(X)
        uniques = np.unique(X.flatten())
        self.player_dict = {player: idx for idx, player in enumerate(uniques)}
        return self

    def transform(self, X):        
        X = np.asarray(X)
        n_samples = X.shape[0]
        n_features = len(self.player_dict)
        rows, cols, data = [], [], []

        for i, row in enumerate(X):
            for j, player in enumerate(row):
                col_idx = self.player_dict.get(player)
                if col_idx is not None:
                    rows.append(i)
                    cols.append(col_idx)
                    data.append(1 if j < len(row)//2 else -1)

        return sparse.csr_matrix((data, (rows, cols)), shape=(n_samples, n_features), dtype=int)

class TeamBagEncoder(BaseEstimator, TransformerMixin):   
    def __init__(self):
        self.team_dict = None

    def fit(self, X, y=None):      
        X = np.asarray(X)
        uniques = np.unique(X.flatten())
        self.team_dict = {team: idx for idx, team in enumerate(uniques)}
        return self

    def transform(self, X):        
        X = np.asarray(X)
        n_samples = X.shape[0]
        n_features = len(self.team_dict)
        rows, cols, data = [], [], []

        for i, row in enumerate(X):            
            for j, team in enumerate(row):
                col_idx = self.team_dict.get(team)
                if col_idx is not None:
                    rows.append(i)
                    cols.append(col_idx)
                    data.append(1 if j == 0 else -1)

        return sparse.csr_matrix((data, (rows, cols)), shape=(n_samples, n_features), dtype=int)

from joblib import Parallel, delayed
from tqdm.notebook import tqdm
from tqdm_joblib import tqdm_joblib
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import shuffle
from scipy import sparse
import numpy as np
from sklearn.metrics import roc_auc_score

class RecursiveTimeSeriesPermutationSelector(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, n_splits=10, n_repeats=1, scoring="roc_auc", random_state=42):
        self.estimator = estimator
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.scoring = scoring  # kept for compatibility but always roc_auc
        self.random_state = random_state
        self.mask_ = None

    def _perm_feature_score(self, est, X_val, y_val, i, rng):
        perm_scores = []
        for _ in range(self.n_repeats):
            X_perm = X_val.toarray() if sparse.issparse(X_val) else X_val.copy()
            seed = rng.randint(0, 2**32 - 1)
            X_perm[:, i] = shuffle(X_perm[:, i], random_state=seed)
            y_pred = est.predict_proba(X_perm)[:, 1]
            perm_scores.append(roc_auc_score(y_val, y_pred))
        return np.mean(perm_scores)

    def fit(self, X, y):
        if sparse.issparse(X):
            X = X.tocsr()
        else:
            X = np.asarray(X)

        mask_all = np.ones(X.shape[1], dtype=bool)
        iteration = 0
        rng = np.random.RandomState(self.random_state)

        while True:
            iteration += 1
            print(f"Iteration {iteration}: {mask_all.sum()} features")
            importances_accum = np.zeros(mask_all.sum())
            X_selected = X[:, mask_all]
            tscv = TimeSeriesSplit(n_splits=self.n_splits)

            for train_idx, val_idx in tscv.split(X_selected):
                X_train, X_val = X_selected[train_idx], X_selected[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                est = clone(self.estimator)
                est.fit(X_train, y_train)
                y_pred_orig = est.predict_proba(X_val)[:, 1]
                score_orig = roc_auc_score(y_val, y_pred_orig)

                # Parallel feature permutation with tqdm
                with tqdm_joblib(tqdm(total=X_selected.shape[1], desc="Permuting features")):
                    perm_scores = Parallel(n_jobs=-1)(
                        delayed(self._perm_feature_score)(est, X_val, y_val, i, rng)
                        for i in range(X_selected.shape[1])
                    )

                importances_accum += score_orig - np.array(perm_scores)

            importances_avg = importances_accum / self.n_splits
            mask_iteration = importances_avg > 0
            if mask_iteration.sum() == mask_all.sum():
                break
            idx_remaining = np.where(mask_all)[0]
            mask_all[idx_remaining[~mask_iteration]] = False

        self.mask_ = mask_all
        print(f"Final selected features: {mask_all.sum()} / {mask_all.size}")
        return self

    def transform(self, X):
        if self.mask_ is None:
            raise RuntimeError("You must fit before calling transform.")
        return X[:, self.mask_] if sparse.issparse(X) else np.asarray(X)[:, self.mask_]

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


# ----------------------- Metrics -----------------------
def get_metrics(y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "auc": auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }
    
import os
import argparse
from dotenv import load_dotenv

def get_settings():
    """
    Parse CLI arguments for .env file, load environment variables, 
    and return settings for the pipeline.
    """
    parser = argparse.ArgumentParser(description="Load ML pipeline settings")
    parser.add_argument(
        "--env_file",
        type=str,
        default=".env",
        help="Path to .env file containing environment variables"
    )
    args = parser.parse_args()

    # Load .env file
    if os.path.exists(args.env_file):
        load_dotenv(args.env_file)
    else:
        print(f"Warning: .env file '{args.env_file}' not found. Using defaults or env vars.")

    # Read settings from environment variables or set defaults
    settings = {
        "PATH_TO_GAMES_RAW_DIR": os.getenv("PATH_TO_GAMES_RAW_DIR", "data/games_raw"),
        "TEST_SIZE": int(os.getenv("TEST_SIZE", 100)),
        "PATH_TO_ML_RESULTS_DIR": os.getenv("PATH_TO_ML_RESULTS_DIR", "data/ml")
    }

    # Ensure results directory exists
    os.makedirs(settings["PATH_TO_ML_RESULTS_DIR"], exist_ok=True)

    return settings

def main():
    # ----------------------- Load settings -----------------------
    settings = get_settings()
    PATH_TO_GAMES_RAW = settings["PATH_TO_GAMES_RAW_DIR"]
    TEST_SIZE = settings["TEST_SIZE"]
    PATH_TO_ML_RESULTS_DIR = settings["PATH_TO_ML_RESULTS_DIR"]

    # ----------------------- Load data -----------------------
    game_ids = get_game_ids(PATH_TO_GAMES_RAW)
    game_ids_train, game_ids_test = game_ids[:-TEST_SIZE], game_ids[-TEST_SIZE:]

    X_train, y_train = get_X_y(PATH_TO_GAMES_RAW, game_ids_train)
    X_test, y_test = get_X_y(PATH_TO_GAMES_RAW, game_ids_test)

    # ----------------------- Pipeline -----------------------
    team_cols = [0, 1]
    player_cols = list(range(2, X_train.shape[1]))

    pipeline = Pipeline([
        ("features", FeatureUnion([
            ("team_bag", Pipeline([
                ("select_teams", ColumnSelector(team_cols)),
                ("team_encoder", TeamBagEncoder())
            ])),
            ("player_bag", Pipeline([
                ("select_players", ColumnSelector(player_cols)),
                ("player_encoder", PlayerBagEncoder())
            ]))
        ])),
        ("feature_selector", RecursiveTimeSeriesPermutationSelector(
            estimator=LogisticRegression(solver="liblinear", max_iter=1000),
            n_splits=10,
            n_repeats=1,
            random_state=42
        )),
        ("logit", LogisticRegression(
            solver="liblinear",
            max_iter=1000,
            random_state=42
        ))
    ])

    # ----------------------- Fit pipeline -----------------------
    pipeline.fit(X_train, y_train)

    # ----------------------- Predict & metrics -----------------------
    y_test_pred = pipeline.predict(X_test)
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = get_metrics(y_test, y_test_pred, y_test_proba)

    # ----------------------- Hash & save -----------------------
    game_ids_bytes = json.dumps(game_ids, sort_keys=True).encode("utf-8")
    game_ids_hash = hashlib.md5(game_ids_bytes).hexdigest()

    json_path = os.path.join(PATH_TO_ML_RESULTS_DIR, f"{game_ids_hash}.json")
    pickle_path = os.path.join(PATH_TO_ML_RESULTS_DIR, f"{game_ids_hash}.pickle")

    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)

    with open(pickle_path, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"Metrics saved to {json_path}")
    print(f"Pipeline saved to {pickle_path}")

if __name__ == "__main__":
    main()

