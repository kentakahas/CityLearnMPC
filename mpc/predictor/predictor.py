import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from mpc.predictor.sequences import generate_sequence
from mpc.predictor.load import load


class BasePredictor:

    def __init__(self, building, feature, lookback, horizon):
        self.building = building
        self.feature = feature
        self.lookback = lookback
        self.horizon = horizon

        self.model = None

        self.package_dir = os.path.dirname(os.path.abspath(__file__))

        data = pd.read_csv(os.path.join(self.package_dir, "..", "data", f"{building}.csv"))
        sequence = generate_sequence(data, lookback, horizon, feature)
        db = load(sequence)

        self.X = db.data
        self.y = db.target
        self.n_features = db.n_features
        self.n_targets = db.n_targets

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        self.directory = os.path.join(self.package_dir, "..", "models", self.building)

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def fit(self, X=None, y=None):
        if X is None and y is None:
            X = self.X_train
            y = self.y_train

        self.model.fit(X, y)

    def predict(self, y):
        return self.model.predict(y)

    def evaluate(self, metric=mean_squared_error, X=None, y=None) -> tuple[float, list]:
        if X and y is None:
            X = self.X_test
            y = self.y_test

        score = []

        y_pred = self.model.predict(X)
        for test, pred in zip(y, y_pred):
            score.append(metric(test, pred))
        return np.mean(score), score

    def save(self, filename: str = None):
        if filename is None:
            filename = f'{self.lookback}_{self.feature}_{self.horizon}.pkl'
        self.model.save(os.path.join(filename))
