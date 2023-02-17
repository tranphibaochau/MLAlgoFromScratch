import numpy as np
from collections import Counter
from DecisionTree import DecisionTree
import random

class AdaBoost(DecisionTree):
    def __init__(self, n_estimators=None, weights=[]):
        self.n_estimators = n_estimators
        self.weights = weights
        self.stumps = []

    def _weighted_boostrap_sampling(self, X, y, weight):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True, p=weight)
        return X[idxs], y[idxs]

    def fit(self, X, y):
        weights = np.ones(X.shape[0]) / X.shape[0]
        self.n_estimators = X.shape[1]

        # create an array of initially equal weight
        for _ in range(self.n_estimators):
            # weighted bootstrap sampling with replacement
            X_sample, y_sample = self._weighted_boostrap_sampling(X, y, weights)
            stump = DecisionTree(max_depth=1, n_features=self.n_estimators, criterion="gini")
            stump.fit(X_sample, y_sample)
            y_pred = stump.predict(X)

            # calculate total error and amount of say of the stump
            total_error = np.sum(weights * (y_pred != y))
            amount_of_say = 0.5 * np.log((1-total_error)/total_error)

            # recalculate the weights of each input
            weights *= np.exp(-amount_of_say * y * y_pred)
            weights /= np.sum(weights)

            # add current stump to the list of stumps along with its amount_of_say
            self.stumps.append(stump)
            self.weights.append(amount_of_say)

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])

        for i in range(self.n_estimators):
            y_pred += self.weights[i] * self.stumps[i].predict(X)

        return np.sign(y_pred)
