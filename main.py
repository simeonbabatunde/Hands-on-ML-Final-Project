#!/usr/bin/env python
import numpy as np
from utils import *
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Initial weights
    np.random.seed(100)
    wts = np.random.rand(1,13)[0]
    alpha = 0.05
    epocs = 1000
    error_epocs = np.zeros([epocs,2])
    weights_array = np.zeros([3,13])            # This saves weights

    dataset = read_file("activity_recognition_dataset.csv")
    train,test = data_preprocess(dataset, 0.8)

    # Loop through the 3 classes by comparing one against others
    # Keep an array of weights for those 3 runs
    # for i in range(3):
    #     # Set label to 1 for corresponding i and others to zero
    #     label = (train[:, -1] == i).astype(int)
    #     train[:, -1] = label
    #     # Compute Error and save in array
    #     error_j = j_error(wts, train)
    #     error_epocs[0][0] = 0
    #     error_epocs[0][1] = error_j
    #
    #     # Train the algorithm for some epocs
    #     for epoc in range(epocs - 1):
    #         wts = gradient_descent(wts, alpha, train)
    #         error_j = j_error(wts, train)
    #         error_epocs[epoc+1][0] = epoc + 1
    #         error_epocs[epoc+1][1] = error_j
    #
    #     for m in range(13):
    #         weights_array[i,m] = wts[m]
    #     plot_error(error_epocs, str(i+1))
    #
    #
    # np.savetxt("weigths.csv", weights_array, delimiter=",")
    #
    weights_arr = np.loadtxt(fname = "weigths.csv", delimiter=',')


    predictions = predict(weights_array, test[:, :-1])

    correct = compute_correction(predictions, test[:, -1])
    print(correct, float(correct)/len(predictions))
