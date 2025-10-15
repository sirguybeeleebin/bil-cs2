import os
import json
import pickle
import argparse
import logging
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from typing import Optional
from tqdm import tqdm

# -----------------------------
# Global logger setup
# -----------------------------
log = logging.getLogger("MLPipeline")
log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
log.addHandler(ch)


class ML:
    def __init__(self, random_state: int = 13):
        self.random_state = random_state
        self.clf: Optional[LogisticRegression] = None
        self.selected_features: Optional[np.ndarray] = None
        self.player_dict: Optional[dict] = None
        self.team_dict: Optional[dict] = None

    # -------------------------
    # Bag-of-players and teams
    # -------------------------
    def _transform_player_bag(self, X_players: np.ndarray) -> csr_matrix:
        n_samples = X_players.shape[0]
        n_features = len(self.player_dict)
        rows, cols, data = [], [], []
        for i in range(n_samples):
            for j in range(5):  # Team1 +1
                idx = self.player_dict.get(X_players[i, j])
                if idx is not None:
                    rows.append(i)
                    cols.append(idx)
                    data.append(1)
            for j in range(5, 10):  # Team2 -1
                idx = self.player_dict.get(X_players[i, j])
                if idx is not None:
                    rows.append(i)
                    cols.append(idx)
                    data.append(-1)
        return csr_matrix((data, (rows, cols)), shape=(n_samples, n_features))

    def _transform_team_bag(self, X_teams: np.ndarray) -> csr_matrix:
        n_samples = X_teams.shape[0]
        n_features = len(self.team_dict)
        rows, cols, data = [], [], []
        for i in range(n_samples):
            t1_idx = self.team_dict.get(X_teams[i, 0])
            t2_idx = self.team_dict.get(X_teams[i, 1])
            if t1_idx is not None:
                rows.append(i)
                cols.append(t1_idx)
                data.append(1)
            if t2_idx is not None:
                rows.append(i)
                cols.append(t2_idx)
                data.append(-1)
        return csr_matrix((data, (rows, cols)), shape=(n_samples, n_features))

    def _transform_bag(self, X_players: np.ndarray, X_teams: np.ndarray) -> csr_matrix:
        return hstack([self._transform_player_bag(X_players), self._transform_team_bag(X_teams)], format='csr')

    # -------------------------
    # Recursive L1 selection
    # -------------------------
    def _recursive_l1_selection(self, X: np.ndarray, y: np.ndarray, C: float = 1.0) -> np.ndarray:
        n_features = X.shape[1]
        selected_mask = np.ones(n_features, dtype=bool)
        X_selected = X.copy()
        iteration = 0
        while True:
            iteration += 1
            clf_l1 = LogisticRegression(solver='liblinear', penalty='l1', C=C, random_state=self.random_state)
            clf_l1.fit(X_selected, y)
            coef_nonzero = (clf_l1.coef_.flatten() != 0)
            new_selected_mask = np.zeros_like(selected_mask)
            new_selected_mask[selected_mask] = coef_nonzero
            log.info("Iteration %d: %d features selected", iteration, coef_nonzero.sum())
            if (new_selected_mask == selected_mask).all():
                log.info("Feature selection converged.")
                break
            selected_mask = new_selected_mask
            X_selected = X[:, selected_mask]
        return selected_mask

    # -------------------------
    # GridSearch optimization
    # -------------------------
    def _optimize_logit(self, X: np.ndarray, y: np.ndarray) -> float:
        param_grid = {"C": np.linspace(0.01, 1, 10)}
        tscv = TimeSeriesSplit(n_splits=10)
        grid = GridSearchCV(
            LogisticRegression(solver='liblinear', random_state=self.random_state),
            param_grid, scoring='roc_auc', cv=tscv, n_jobs=-1
        )
        grid.fit(X, y)
        best_C = grid.best_params_['C']
        log.info("Best C found: %.4f (AUC=%.4f)", best_C, grid.best_score_)
        return best_C

    # -------------------------
    # Fit / Predict
    # -------------------------
    def fit(self, X_players: np.ndarray, X_teams: np.ndarray, y: np.ndarray) -> "ML":
        log.info("Starting recursive L1 feature selection...")
        self.player_dict = {pid: idx for idx, pid in enumerate(np.unique(X_players))}
        self.team_dict = {tid: idx for idx, tid in enumerate(np.unique(X_teams))}
        X_encoded = self._transform_bag(X_players, X_teams)
        self.selected_features = self._recursive_l1_selection(X_encoded, y)
        X_selected = X_encoded[:, self.selected_features]
        best_C = self._optimize_logit(X_selected, y)
        final_mask = self._recursive_l1_selection(X_selected, y, C=best_C)
        full_selected_mask = np.zeros(X_encoded.shape[1], dtype=bool)
        full_selected_mask[self.selected_features] = final_mask
        self.selected_features = full_selected_mask
        self.clf = LogisticRegression(solver='liblinear', C=best_C, random_state=self.random_state)
        self.clf.fit(X_encoded[:, self.selected_features], y)
        log.info("Number of selected features: %d", self.selected_features.sum())
        return self

    def predict_proba(self, X_players: np.ndarray, X_teams: np.ndarray) -> np.ndarray:
        X_encoded = self._transform_bag(X_players, X_teams)
        X_selected = X_encoded[:, self.selected_features]
        return self.clf.predict_proba(X_selected)[:, 1]

    def predict(self, X_players: np.ndarray, X_teams: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X_players, X_teams) >= threshold).astype(int)

    


# -----------------------------
# Dataset functions
# -----------------------------
def get_X_players(game_ids, path_to_games_raw_dir):
    X = []
    for game_id in tqdm(game_ids):        
        with open(os.path.join(path_to_games_raw_dir, f"{game_id}.json"), "r", encoding="utf-8") as f:
            game_data = json.load(f)            
        dd = defaultdict(list)
        for p in game_data["players"]:
            dd[p["team"]["id"]].append(p["player"]["id"])        
        t1_id, t2_id = sorted(dd.keys())
        team1_players = sorted(dd[t1_id])
        team2_players = sorted(dd[t2_id])
        X.append(team1_players + team2_players)
    return np.array(X)


def get_X_teams(game_ids, path_to_games_raw_dir):
    X = []
    for game_id in tqdm(game_ids):        
        with open(os.path.join(path_to_games_raw_dir, f"{game_id}.json"), "r", encoding="utf-8") as f:
            game_data = json.load(f)            
        dd = defaultdict(list)
        for p in game_data["players"]:
            dd[p["team"]["id"]].append(p["player"]["id"])        
        t1_id, t2_id = sorted(dd.keys())
        X.append([t1_id, t2_id])
    return np.array(X)

def get_info(game_ids, path_to_games_raw_dir):
    """
    Extracts game info: map_id, league_id, serie_id, tournament_id, serie_tier
    """
    info_list = []
    for game_id in tqdm(game_ids):
        with open(os.path.join(path_to_games_raw_dir, f"{game_id}.json"), "r", encoding="utf-8") as f:
            game_data = json.load(f)
        
        map_id = game_data.get("map", {}).get("id", None)
        match = game_data.get("match", {})
        league_id = match.get("league", {}).get("id", None)
        serie_id = match.get("serie", {}).get("id", None)
        tournament_id = match.get("tournament", {}).get("id", None)
        serie_tier = match.get("serie", {}).get("tier", None)
        
        info_list.append([map_id, league_id, serie_id, tournament_id, serie_tier])
    
    return np.array(info_list)


def get_y(game_ids, path_to_games_raw_dir):
    y = []
    for game_id in tqdm(game_ids):
        with open(os.path.join(path_to_games_raw_dir, f"{game_id}.json"), "r", encoding="utf-8") as f:
            game_data = json.load(f)
        dd = defaultdict(list)
        for p in game_data["players"]:
            dd[p["team"]["id"]].append(p["player"]["id"])
        t1_id, t2_id = sorted(dd.keys())
        winner_counts = defaultdict(int)
        for round_info in game_data["rounds"]:
            winner = round_info.get("winner_team")
            if winner is not None:
                winner_counts[winner] += 1
        winner_team = max(winner_counts, key=winner_counts.get)
        y.append(int(winner_team == t1_id))
    return np.array(y)


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train ML model on train/test game IDs.")
    parser.add_argument("--path_to_games_raw_dir", type=str, default="data/games_raw")
    parser.add_argument("--path_to_train_test_split_file", type=str, default="data/train_test_splits/8eaa28297645dca5.json")
    parser.add_argument("--path_to_ml_results_dir", type=str, default="data/ml_results")    
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.path_to_ml_results_dir, exist_ok=True)

    log.info("Loading train/test game IDs...")
    with open(args.path_to_train_test_split_file, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    train_ids = split_data["train"]
    test_ids = split_data["test"]

    hash_id = os.path.splitext(os.path.basename(args.path_to_train_test_split_file))[0]

    log.info("Initializing ML pipeline...")
    pipeline = ML()

    # -----------------------------
    # Build train dataset
    # -----------------------------
    log.info("Building train dataset...")
    X_train_players = get_X_players(train_ids, args.path_to_games_raw_dir)
    X_train_teams = get_X_teams(train_ids, args.path_to_games_raw_dir)
    y_train = get_y(train_ids, args.path_to_games_raw_dir)

    # -----------------------------
    # Build test dataset
    # -----------------------------
    log.info("Building test dataset...")
    X_test_players = get_X_players(test_ids, args.path_to_games_raw_dir)
    X_test_teams = get_X_teams(test_ids, args.path_to_games_raw_dir)
    y_test = get_y(test_ids, args.path_to_games_raw_dir)

    # -----------------------------
    # Train model
    # -----------------------------
    log.info("Training model...")
    pipeline.fit(X_train_players, X_train_teams, y_train)

    # -----------------------------
    # Predict and evaluate
    # -----------------------------
    log.info("Predicting on test set...")
    y_pred = pipeline.predict(X_test_players, X_test_teams)
    y_proba = pipeline.predict_proba(X_test_players, X_test_teams)

    tn = np.sum((y_test == 0) & (y_pred == 0))
    fp = np.sum((y_test == 0) & (y_pred == 1))
    fn = np.sum((y_test == 1) & (y_pred == 0))
    tp = np.sum((y_test == 1) & (y_pred == 1))

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 2),
        "precision": round(float(precision_score(y_test, y_pred)), 2),
        "recall": round(float(recall_score(y_test, y_pred)), 2),
        "f1_score": round(float(f1_score(y_test, y_pred)), 2),
        "auc": round(float(roc_auc_score(y_test, y_proba)), 2),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "n_train": len(y_train),
        "n_test": len(y_test)
    }

    log.info("Metrics computed:\n%s", json.dumps(metrics, indent=4))

    # -----------------------------
    # Save model and metrics
    # -----------------------------
    model_file = os.path.join(args.path_to_ml_results_dir, f"{hash_id}.pickle")
    with open(model_file, "wb") as f:
        pickle.dump(pipeline, f)
    log.info("Model saved to %s", model_file)

    metrics_file = os.path.join(args.path_to_ml_results_dir, f"{hash_id}.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    log.info("Metrics saved to %s", metrics_file)


if __name__ == "__main__":
    main()
