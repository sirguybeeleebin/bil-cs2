import optuna
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection._split import BaseCrossValidator

class LogitOptimizer:
    def __init__(self, cv: BaseCrossValidator, n_trials: int = 50, random_state: int | None = None):
        self.cv: BaseCrossValidator = cv
        self.n_trials: int = n_trials
        self.random_state: int | None = random_state
        self.best_model_: LogisticRegression | None = None
        self._best_params: dict | None = None

    def _objective(self, trial, X: np.ndarray, y: np.ndarray) -> float:
        C = trial.suggest_float("C", 0.01, 10.0, log=True)
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
        solver = "liblinear" if penalty == "l1" else "lbfgs"

        model = LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            max_iter=1000,
            random_state=self.random_state
        )

        scores = cross_val_score(model, X, y, cv=self.cv, scoring="roc_auc")
        return 1.0 - np.mean(scores)  # minimize 1 - AUC

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogitOptimizer":
        def objective(trial):
            return self._objective(trial, X, y)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)
        self._best_params = study.best_params        
        return self

    def get_best_params(self) -> dict:
        if self._best_params is None:
            raise ValueError("Call fit() first")
        return self._best_params

    
