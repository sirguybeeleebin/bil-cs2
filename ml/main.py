import argparse
import json
import logging
import os
import pickle
from datetime import datetime

import clickhouse_connect
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ml_stacking_pipeline")


# ---------------------------
# Player Pipeline
# ---------------------------
class PlayerBagPipeline:
    def __init__(self, n_folds=5, random_state=13):
        self._player_label_dict = None
        self._n_players = None
        self.n_folds = n_folds
        self.random_state = random_state
        self._logits = []

    def fit(self, X, y):
        ids = np.unique(X.flatten())
        self._player_label_dict = dict(zip(ids, range(len(ids))))
        self._n_players = len(ids)
        X_bag = self._transform(X)

        self._logits = []
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        oof_proba = np.zeros(len(y), dtype=np.float32)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_bag, y)):
            log.info(f"Player fold {fold + 1}/{self.n_folds}")
            model = LogisticRegression(
                random_state=self.random_state, solver="liblinear"
            )
            model.fit(X_bag[train_idx], y[train_idx])
            oof_proba[val_idx] = model.predict_proba(X_bag[val_idx])[:, 1]
            self._logits.append(model)
        return oof_proba

    def _transform(self, X):
        bag = np.zeros((X.shape[0], self._n_players), dtype=np.int8)
        for i in range(X.shape[0]):
            for j in range(10):
                idx = self._player_label_dict.get(X[i, j])
                if idx is None:
                    continue
                bag[i, idx] = 1 if j <= 4 else -1
        return bag

    def predict_proba(self, X):
        X_bag = self._transform(X)
        proba = np.mean(
            [model.predict_proba(X_bag)[:, 1] for model in self._logits], axis=0
        )
        return proba


# ---------------------------
# Team Pipeline
# ---------------------------
class TeamBagPipeline:
    def __init__(self, n_folds=5, random_state=13):
        self._team_label_dict = None
        self._n_teams = None
        self.n_folds = n_folds
        self.random_state = random_state
        self._logits = []

    def fit(self, X, y):
        unique_teams = np.unique(X)
        self._team_label_dict = dict(zip(unique_teams, range(len(unique_teams))))
        self._n_teams = len(unique_teams)
        X_bag = self._transform(X)

        self._logits = []
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        oof_proba = np.zeros(len(y), dtype=np.float32)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_bag, y)):
            log.info(f"Team fold {fold + 1}/{self.n_folds}")
            model = LogisticRegression(
                random_state=self.random_state, solver="liblinear"
            )
            model.fit(X_bag[train_idx], y[train_idx])
            oof_proba[val_idx] = model.predict_proba(X_bag[val_idx])[:, 1]
            self._logits.append(model)
        return oof_proba

    def _transform(self, X):
        bag = np.zeros((X.shape[0], self._n_teams), dtype=np.int8)
        for i in range(X.shape[0]):
            team1, team2 = X[i]
            idx1 = self._team_label_dict.get(team1)
            idx2 = self._team_label_dict.get(team2)
            if idx1 is not None:
                bag[i, idx1] = 1
            if idx2 is not None:
                bag[i, idx2] = -1
        return bag

    def predict_proba(self, X):
        X_bag = self._transform(X)
        proba = np.mean(
            [model.predict_proba(X_bag)[:, 1] for model in self._logits], axis=0
        )
        return proba


# ---------------------------
# Stacking Pipeline
# ---------------------------
class MLStackingPipeline:
    def __init__(self, n_folds=5, random_state=13):
        self.player_pipeline = PlayerBagPipeline(
            n_folds=n_folds, random_state=random_state
        )
        self.team_pipeline = TeamBagPipeline(n_folds=n_folds, random_state=random_state)
        self.meta_model = LogisticRegression(
            random_state=random_state, solver="liblinear"
        )

    def fit(self, X_player, X_team, y):
        log.info("Fitting player pipeline...")
        player_oof = self.player_pipeline.fit(X_player, y)
        log.info("Fitting team pipeline...")
        team_oof = self.team_pipeline.fit(X_team, y)

        meta_features = np.vstack([player_oof, team_oof]).T
        log.info("Training meta-model...")
        self.meta_model.fit(meta_features, y)
        return player_oof, team_oof

    def predict_proba(self, X_player, X_team):
        player_proba = self.player_pipeline.predict_proba(X_player)
        team_proba = self.team_pipeline.predict_proba(X_team)
        meta_features = np.vstack([player_proba, team_proba]).T
        return self.meta_model.predict_proba(meta_features)[:, 1]

    def predict(self, X_player, X_team):
        return np.round(self.predict_proba(X_player, X_team)).astype(int)


# ---------------------------
# ClickHouse data loading
# ---------------------------
def load_player_team_data(client, db, table, game_ids):
    if not game_ids:
        return np.array([]), np.array([]), np.array([])

    ids_str = ",".join(map(str, game_ids))
    query = f"""
        WITH games_agg AS (
            SELECT
                game_id,
                team_id,
                sum(round_win) AS rounds_won,
                arraySort(groupArray(player_id)) AS players_sorted
            FROM {db}.{table}
            WHERE game_id IN ({ids_str})
            GROUP BY game_id, team_id
        ),
        teams AS (
            SELECT
                game_id,
                min(team_id) AS team1,
                max(team_id) AS team2
            FROM games_agg
            GROUP BY game_id
        ),
        winners AS (
            SELECT
                game_id,
                team_id AS winner_team
            FROM games_agg
            QUALIFY ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY rounds_won DESC) = 1
        )
        SELECT
            t.game_id,
            arrayConcat(arraySlice(ga1.players_sorted, 1, 5),
                        arraySlice(ga2.players_sorted, 1, 5)) AS X_player,
            (t.team1, t.team2) AS X_team,
            if(winner_team = t.team1, 1, 0) AS y
        FROM teams AS t
        INNER JOIN winners AS w USING (game_id)
        INNER JOIN games_agg AS ga1 ON ga1.game_id = t.game_id AND ga1.team_id = t.team1
        INNER JOIN games_agg AS ga2 ON ga2.game_id = t.game_id AND ga2.team_id = t.team2
    """
    df = client.query_df(query)
    X_player = np.stack(df["X_player"].values)
    X_team = np.stack(df["X_team"].values)
    y = df["y"].to_numpy(dtype=np.int64)
    log.info(f"Loaded {len(y)} games from ClickHouse")
    return X_player, X_team, y


# ---------------------------
# Main function
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MLStackingPipeline from ClickHouse"
    )
    parser.add_argument("--clickhouse_host", type=str, default="localhost")
    parser.add_argument("--clickhouse_port", type=int, default=8123)
    parser.add_argument("--clickhouse_user", type=str, default="cs2_user")
    parser.add_argument("--clickhouse_password", type=str, default="cs2_password")
    parser.add_argument("--clickhouse_db", type=str, default="cs2_db")
    parser.add_argument("--table_name", type=str, default="games_flat")
    parser.add_argument(
        "--train_test_split_file",
        type=str,
        default="data/train_test_splits/238ce5b0a5f7b8c443ad022b137a7311.json",
    )
    parser.add_argument("--ml_results_dir", type=str, default="data/ml")
    parser.add_argument("--n_folds", type=int, default=10)
    parser.add_argument("--random_state", type=int, default=13)
    return parser.parse_args()


def round_metrics(metrics):
    return {k: round(v, 2) for k, v in metrics.items()}


def main():
    args = parse_args()

    log.info("Connecting to ClickHouse...")
    client = clickhouse_connect.get_client(
        host=args.clickhouse_host,
        port=args.clickhouse_port,
        username=args.clickhouse_user,
        password=args.clickhouse_password,
        database=args.clickhouse_db,
    )
    log.info(
        f"✅ Connected to ClickHouse at {args.clickhouse_host}:{args.clickhouse_port}"
    )

    with open(args.train_test_split_file, "r") as f:
        split = json.load(f)
    train_ids = split.get("train", [])
    test_ids = split.get("test", [])
    log.info(f"Train games: {len(train_ids)}, Test games: {len(test_ids)}")

    log.info("Loading train data...")
    X_player_train, X_team_train, y_train = load_player_team_data(
        client, args.clickhouse_db, args.table_name, train_ids
    )
    log.info(
        f"Train shapes: X_player={X_player_train.shape}, X_team={X_team_train.shape}, y={y_train.shape}"
    )

    log.info("Loading test data...")
    X_player_test, X_team_test, y_test = load_player_team_data(
        client, args.clickhouse_db, args.table_name, test_ids
    )
    log.info(
        f"Test shapes: X_player={X_player_test.shape}, X_team={X_team_test.shape}, y={y_test.shape}"
    )

    log.info("Training stacking pipeline...")
    pipeline = MLStackingPipeline(n_folds=args.n_folds, random_state=args.random_state)
    pipeline.fit(X_player_train, X_team_train, y_train)

    # ---------------------------
    # Base model predictions and metrics
    # ---------------------------
    player_proba_test = pipeline.player_pipeline.predict_proba(X_player_test)
    player_pred_test = np.round(player_proba_test).astype(int)

    team_proba_test = pipeline.team_pipeline.predict_proba(X_team_test)
    team_pred_test = np.round(team_proba_test).astype(int)

    player_metrics = round_metrics(
        {
            "f1_score": float(f1_score(y_test, player_pred_test)),
            "auc": float(roc_auc_score(y_test, player_proba_test)),
            "precision": float(precision_score(y_test, player_pred_test)),
            "recall": float(recall_score(y_test, player_pred_test)),
        }
    )

    team_metrics = round_metrics(
        {
            "f1_score": float(f1_score(y_test, team_pred_test)),
            "auc": float(roc_auc_score(y_test, team_proba_test)),
            "precision": float(precision_score(y_test, team_pred_test)),
            "recall": float(recall_score(y_test, team_pred_test)),
        }
    )

    # ---------------------------
    # Meta-model metrics
    # ---------------------------
    y_proba = pipeline.predict_proba(X_player_test, X_team_test)
    y_pred = pipeline.predict(X_player_test, X_team_test)

    meta_metrics = round_metrics(
        {
            "f1_score": float(f1_score(y_test, y_pred)),
            "auc": float(roc_auc_score(y_test, y_proba)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
        }
    )

    # ---------------------------
    # Save all coefficients
    # ---------------------------
    player_coefs = [model.coef_.tolist() for model in pipeline.player_pipeline._logits]
    team_coefs = [model.coef_.tolist() for model in pipeline.team_pipeline._logits]
    meta_coefs = pipeline.meta_model.coef_.tolist()

    metrics = {
        "n_train_samples": int(len(X_player_train)),
        "n_test_samples": int(len(X_player_test)),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "player_model": {"metrics": player_metrics, "coefs": player_coefs},
        "team_model": {"metrics": team_metrics, "coefs": team_coefs},
        "meta_model": {"metrics": meta_metrics, "coefs": meta_coefs},
    }

    os.makedirs(args.ml_results_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.train_test_split_file))[0]

    metrics_path = os.path.join(args.ml_results_dir, f"{base_name}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    log.info(f"✅ Saved metrics to: {metrics_path}")

    model_path = os.path.join(args.ml_results_dir, f"{base_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"pipeline": pipeline}, f)
    log.info(f"✅ Saved stacked model to: {model_path}")

    metrics_dict = {
        "Model": ["Player", "Team", "Meta"],
        "F1 Score": [
            metrics["player_model"]["metrics"]["f1_score"],
            metrics["team_model"]["metrics"]["f1_score"],
            metrics["meta_model"]["metrics"]["f1_score"],
        ],
        "AUC": [
            metrics["player_model"]["metrics"]["auc"],
            metrics["team_model"]["metrics"]["auc"],
            metrics["meta_model"]["metrics"]["auc"],
        ],
        "Precision": [
            metrics["player_model"]["metrics"]["precision"],
            metrics["team_model"]["metrics"]["precision"],
            metrics["meta_model"]["metrics"]["precision"],
        ],
        "Recall": [
            metrics["player_model"]["metrics"]["recall"],
            metrics["team_model"]["metrics"]["recall"],
            metrics["meta_model"]["metrics"]["recall"],
        ],
    }

    df_metrics = pd.DataFrame(metrics_dict)
    print("\n📊 Model Metrics Table:")
    print(df_metrics)
    log.info(f"✅ Done at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
