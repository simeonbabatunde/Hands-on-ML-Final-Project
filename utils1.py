import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

'''
Useful Types
'''
class AccuracyStats:
    def __init__(self):
        self.accuracy = 0
        self.f_0 = 0
        self.f_1 = 0
        self.f_10 = 0
        self.f_01 = 0
        self.precision = 0
        self.recall = 0


'''
Helpful functions for data preprocessing, training and testing
'''

def read_file(input_file):
    '''
    input_file : string
                Data file name to be loaded
    return : numpy array
             m x n array representing the data loaded, with class column encoded in
             numerical values i.e. 0,1,2 for sitting,standing and walking respectively.
    '''
    try:
        with open(input_file, "r") as fileObject:

            nrows = int(len(fileObject.readlines())) - 1
            # Reset file pointer to the beginning
            fileObject.seek(0)
            # Skip the header
            line = fileObject.readline()
            ncols = 0
            #Create a 3D array filled with zeros
            data = np.zeros([nrows, 13])
            for row in range(nrows):
                line = fileObject.readline()
                line = line.replace('"', '')
                cols = line.strip().split(",")
                if cols[-1] == "sitting":
                    cols[-1] = 0.0
                elif cols[-1] == "standing":
                    cols[-1] = 1.0
                elif cols[-1] == "walking":
                    cols[-1] = 2.0
                else:
                    continue
                # Update numpy array with input data
                ncols = len(cols)
                for col in range(ncols):
                    data[row, col] = float(cols[col])

            return data
    except Exception as e:
        print(e)
        exit(0)


def standardize(array):
    '''
    Description:
        This function Normalizes the dataset using standard mean normalization process i.e. (x-mean)/std
        and returns normalized, mean and std. This feature scaling helps the to speed up convergence of
        cost function J(w).
    Returns:
        array of normalized data
        mean
        std
    '''
    standardized = array
    mean = array.mean(axis=0)
    std = array.std(axis=0)
    rows,cols = array.shape

    for i in range(rows):
        for j in range(cols - 1):
            standardized[i][j] = (array[i][j] - mean[j])/std[j]

    return standardized


def data_preprocess(input_data, train_percent):
    '''
    input_data : numpy array
                m x n numpy data array and ratio of training data
    return : tuple
                (train, test) training data and test data tuple
                each as a numpy array
    '''
    m,n = 0,0
    try:
        # Normalize the dataset using mean-std normalization
        standardized_data = standardize(input_data)
        # Randomize the dataset
        np.random.seed(500)
        np.random.shuffle(standardized_data)
        m,n = standardized_data.shape
        #Split data into training and test
        train, test = standardized_data[:int(m*train_percent), :], standardized_data[int(m*train_percent):, :]

        return (train, test)

    except Exception as e:
        print(e)
        exit(0)


def print_stats(data):
    '''
    data: numpy array
    '''
    pass

def get_accuracy_statistics(actual, pred):
    '''
    actual : list of actual classes
    pred : list of predicted classes
    return : AccuracyStats instance
    '''
    print("======= Fetching Stats ===========")
    print(len(actual) == len(pred))
    accStat = AccuracyStats()
    totalMissed = 0
    total = len(actual)
    for i in range(total):
        p_a = actual[i]
        p_p = pred[i]
        if p_a == p_p:
            if p_a == 1:
                accStat.f_1 += 1
            else:
                accStat.f_0 += 1
        else:
            totalMissed += 1
            if p_a == 1:
                accStat.f_10 += 1
            else:
                accStat.f_01 += 1
    accStat.accuracy = 1 - (totalMissed / total)
    accStat.precision = accStat.f_0 / (accStat.f_0 + accStat.f_10)
    accStat.recall = accStat.f_0 / (accStat.f_0 + accStat.f_01)
    return accStat


def h_x(wts, ftrs):
    '''
    This is the polynomial hypothesis function used by the ML algorithm to predict
    activity based on accelerometer readings.
    Args: Features and weights
    Returns: prediction output (0 to 1)
    '''
    z = wts[0]
    for i in range(len(ftrs)):
        z += wts[i+1] * ftrs[i]

    return 1.0/(1 + np.exp(-z))


def j_error(weights, training_data):
    '''
    This is the Cost/Error function
    Args: Weights and 13-column, training data (x1 . . . x12, y)
    Returns error cost
    '''
    num_rows = len(training_data)
    sum = 0.0

    for i in range(num_rows):
        h_of_x = h_x(weights, training_data[i, :-1])
        sum+= (((training_data[i, -1] * np.log(h_of_x)) + ((1.0 - training_data[i, -1]) * np.log(1.0 - h_of_x))))

    return -sum/num_rows


def partial_derivatives(wts, train_data):
    '''
    This computes the pertial derivatives of the cost function w.r.t the w's
    Args: Weights and training data (x1 . . .x12, y)
    Returns: Array of size 13, containing partial derivative values for w_0 to w_12
    '''
    derivative_array = np.zeros(len(wts))
    sum = np.zeros(len(wts))
    rows = len(train_data)

    for j in range(rows):
        sum[0] += (h_x(wts, train_data[j, :-1]) - train_data[j, -1])
        for k in range(len(wts) - 1):
            sum[k+1] += ((h_x(wts, train_data[j, :-1]) - train_data[j, -1]) * train_data[j,k])

    for l in range(len(wts)):
        derivative_array[l] = sum[l]/rows

    return derivative_array


def gradient_descent(wts, alpha, train_data):
    '''
    This function computes the gradient descent i.e. new values of w's to minimize the
    cost function.
    Args: Initial weigths, learning rate (alpha) and the train data
    Returns: New weights w0-to-w5
    '''
    temp = np.zeros(len(wts))

    derivatives = partial_derivatives(wts, train_data)
    for x in range(len(wts)):
        temp[x] = wts[x] - (alpha * derivatives[x])

    return temp


# Plot of error against epochs
def plot_error(dataframe, plotname):
    plt.figure(figsize=(12, 8))
    for row in range(len(dataframe)):
        plt.scatter(dataframe[row, 0], dataframe[row,1], color="green", marker="o")

    plt.xlabel("Epochs")
    plt.ylabel("Error J(w)")
    plt.title("Plot of Epochs against Error J")
    plt.savefig(plotname, bbox_inches="tight")
    # plt.show()


# Return array of predictions for test data
def predict(wtts, test_data):
    '''
    This function takes in waights and features and generate predictions
    Args: Array of weigths and input features
    Retrurns: Numpy array of predictions 0,1,2 based on the class
    '''
    rows = len(test_data)
    pred_output = np.zeros([3,rows])

    for k in range(rows):
        pred_output[0,k] = h_x(wtts[0], test_data[k, :])
        pred_output[1,k] = h_x(wtts[1], test_data[k, :])
        pred_output[2,k] = h_x(wtts[2], test_data[k, :])

    max_pred = pred_output.max(axis=0)
    pred_results = max_pred


    for s in range(rows):
        if pred_output[0,s] == max_pred[s]:
            pred_results[s] = 0.0
        elif pred_output[1,s] == max_pred[s]:
            pred_results[s] = 1.0
        elif pred_output[2,s] == max_pred[s]:
            pred_results[s] = 2.0


    return pred_results


## This compute error between the prediction and test labels
def compute_score(prediction_array, test_labels):
    correct = np.count_nonzero(prediction_array==test_labels)
    return float(correct)/len(test_labels)


def confusion_matrix(prediction, label):
    y_actu = pd.Series(label, name='Actual')
    y_pred = pd.Series(prediction, name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred)
    return df_confusion

def print_accuracy_stats(pred, actual):
    '''
    actual : list of actual classes
    pred : list of predicted classes
    Prints Accuracy Info (Accuraycy, F1 score and Recall and Precision)
    '''
    print("===================================================================")
    print("Total Correct")
    print(np.count_nonzero(pred == actual))
    print("===================================================================")
    print(classification_report(pred, actual))
    print("===================================================================")
    print(confusion_matrix(pred, actual))
    print("===================================================================")
