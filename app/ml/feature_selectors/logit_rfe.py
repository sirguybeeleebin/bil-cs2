from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import make_scorer, precision_score

class LogitRFECV(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        C: float = 1.0,
        cv=TimeSeriesSplit(n_splits=5),
        scoring=precision_score,
        step: int = 1,
        random_state: int = 42
    ):
        """
        C: коэффициент регуляризации для LogisticRegression
        cv: cross-validation splitter
        scoring: функция метрики или строка
        step: количество признаков, удаляемых на каждой итерации
        random_state: для воспроизводимости
        """
        self.C = C
        self.cv = cv
        self.scoring = scoring
        self.step = step
        self.random_state = random_state
        self.selector_: RFECV = None

    def fit(self, X, y):
        estimator = LogisticRegression(            
            solver='liblinear',
            C=self.C,
            random_state=self.random_state
        )
        scorer = self.scoring
        if callable(self.scoring):
            scorer = make_scorer(self.scoring)

        self.selector_ = RFECV(
            estimator=estimator,
            step=self.step,
            cv=self.cv,
            scoring=scorer,
            n_jobs=-1
        )
        self.selector_.fit(X, y)
        return self

    def transform(self, X):
        if self.selector_ is None:
            raise ValueError("You must fit the selector before transforming.")
        return self.selector_.transform(X)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
