import numpy as np
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