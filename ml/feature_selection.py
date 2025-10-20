import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from tqdm import tqdm


def select_features_with_logit_and_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    Cs: np.ndarray = np.array([0.08, 0.09, 0.1, 0.2, 0.3]),
    scoring: str = "roc_auc",
    random_state: int = 42,
    cv: TimeSeriesSplit = TimeSeriesSplit(10),
    verbose: int = 2,
    n_jobs: int = 1,
) -> np.ndarray:
    scores = []
    masks_per_C = []
    for C in tqdm(Cs):
        logit = LogisticRegression(
            C=C,
            penalty="l1",
            solver="liblinear",
            random_state=random_state,
            max_iter=1000,
        )
        fold_masks = []
        for tr_idx, _ in cv.split(X_train, y_train):
            logit.fit(X_train[tr_idx], y_train[tr_idx])
            fold_masks.append(logit.coef_.flatten() != 0)
        mask_C = np.mean(fold_masks, axis=0) > 0.5
        masks_per_C.append(mask_C)

        score = cross_val_score(
            LogisticRegression(random_state=random_state, solver="liblinear"),
            X_train[:, mask_C],
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
        ).mean()
        scores.append(score)

    best_idx = np.argmax(scores)
    best_mask = masks_per_C[best_idx]

    logit_final = LogisticRegression(
        random_state=random_state, solver="liblinear", max_iter=1000
    )
    X_train_selected = X_train[:, best_mask]

    rfecv = RFECV(
        estimator=logit_final,
        step=1,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    rfecv.fit(X_train_selected, y_train)

    final_mask = np.zeros_like(best_mask, dtype=bool)
    final_mask[best_mask] = rfecv.support_

    return final_mask
