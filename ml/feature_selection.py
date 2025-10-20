from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
import numpy as np
from tqdm import tqdm

def select_features_with_logit_and_cv(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    Cs: np.ndarray = np.linspace(0.00001, 0.01, 100), 
    scoring: str = "roc_auc", 
    random_state: int = 42, 
    cv: TimeSeriesSplit = TimeSeriesSplit(10),
    verbose: int = 2,
) -> np.ndarray:    
    scores: list[float] = []
    
    for C in tqdm(Cs):
        logit = LogisticRegression(C=C, random_state=random_state, solver="liblinear")
        masks: list[np.ndarray] = []        
        for tr_idx, _ in cv.split(X_train, y_train):
            logit.fit(X_train[tr_idx], y_train[tr_idx])
            masks.append(logit.coef_.flatten() != 0)
        mask: np.ndarray = np.mean(masks, axis=0) > 0.5  
        score: float = np.mean(
            cross_val_score(
                logit,
                X_train[:, mask],
                y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
            )
        )
        scores.append(score)

    best_C: float = Cs[np.argmax(scores)]
    logit = LogisticRegression(C=best_C, random_state=random_state, solver="liblinear")
    masks = []
    for tr_idx, _ in cv.split(X_train, y_train):
        logit.fit(X_train[tr_idx], y_train[tr_idx])
        masks.append(logit.coef_.flatten() != 0)
    mask = np.mean(masks, axis=0) > 0.5

    X_train_reduced: np.ndarray = X_train[:, mask]

    rfecv = RFECV(
        estimator=logit,
        step=1,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=verbose,
    )
    rfecv.fit(X_train_reduced, y_train)

    temp_mask: np.ndarray = np.zeros_like(mask, dtype=bool)
    temp_mask[mask] = rfecv.support_
    mask = temp_mask
    return mask
