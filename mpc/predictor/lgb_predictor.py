import os
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.multioutput import MultiOutputRegressor
import joblib
import multiprocessing
from predictor import BasePredictor


class LGBPredictor(BasePredictor):
    def __init__(self, building, feature, lookback, horizon):
        super().__init__(building, feature, lookback, horizon)
        self.lgm_regressor = lgb.LGBMRegressor(verbose=-1, n_jobs=1)
        self.bst = MultiOutputRegressor(self.lgm_regressor)

        self.param_dist = {
            'estimator__n_estimators': randint(50, 500),
            'estimator__max_depth': randint(2, 15),
            'estimator__learning_rate': [0.01, 0.05, 0.1, 0.3, 0.5],
            'estimator__num_leaves': randint(5, 30)
        }

    def search(self, X=None, y=None, n_iter=200, n_jobs=None):
        if X is None and y is None:
            X = self.X_train
            y = self.y_train

        if n_jobs is None:
            n_jobs = multiprocessing.cpu_count()
        else:
            assert n_jobs <= multiprocessing.cpu_count()

        random_search = RandomizedSearchCV(
            self.bst,
            param_distributions=self.param_dist,
            n_iter=n_iter,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=n_jobs,
            verbose=1
        )

        random_search.fit(X, y)
        self.model = random_search.best_estimator_

    def save(self, filename: str = None, file_type="pkl"):
        if filename is None:
            filename = f'{self.lookback}_{self.feature}_{self.horizon}'
        joblib.dump(self.model, os.path.join(self.directory, f"{filename}.{file_type}"))
