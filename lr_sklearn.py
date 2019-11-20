#!/usr/bin/env python
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression




if __name__ == '__main__':
    # Initial weights
    np.random.seed(100)
    wts = np.random.rand(1,13)[0]
    alpha = 0.05
    epocs = 800
    error_epocs = np.zeros([epocs,2])

    dataset = read_file("activity_recognition_dataset.csv")
    train,test = data_preprocess(dataset, 0.8)

    regobject = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr', max_iter=800).fit(train[:, :-1], train[:, -1])
    prediction = regobject.predict(test[:, :-1])

    print(compute_score(prediction, test[:, -1]))
    print_accuracy_stats(prediction, test[:, -1])
