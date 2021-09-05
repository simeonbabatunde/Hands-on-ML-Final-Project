#!/usr/bin/env python
import numpy as np
from utils import *
import matplotlib.pyplot as plt
#https://kenzotakahashi.github.io/logistic-regression-from-scratch-in-python.html



class LogisticRegressionOVR(object):
    def __init__(self, eta=0.1, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _predict_one(self, x):
        return max((x.dot(w), c) for w, c in self.w)[1]

    def predict(self, X):
        return [self._predict_one(i) for i in np.insert(X, 0, 1, axis=1)]

    def score(self, X, y):
        return sum(X == y) / len(y)

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.w = []
        m = X.shape[0]

        for i in np.unique(y):
            y_copy = np.where(y == i, 1, 0)
            w = np.ones(X.shape[1])

            for _ in range(self.n_iter):
                output = X.dot(w)
                errors = y_copy - self._sigmoid(output)
                w += self.eta / m * errors.dot(X)
            self.w.append((w, i))
        return self


if __name__ == '__main__':
    # Initial weights
    np.random.seed(100)
    wts = np.random.rand(1,13)[0]
    alpha = 0.05
    epocs = 800
    error_epocs = np.zeros([epocs,2])

    dataset = read_file("activity_recognition_dataset.csv")
    train,test = data_preprocess(dataset, 0.8)

    multiclass = LogisticRegressionOVR(eta=alpha, n_iter=epocs).fit(train[:, :-1], train[:, -1])
    prediction = multiclass.predict(test[:, :-1])

    print(compute_score(prediction, test[:, -1]))
    print_accuracy_stats(prediction, test[:, -1])
