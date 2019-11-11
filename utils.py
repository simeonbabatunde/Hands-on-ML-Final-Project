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
             m x n array representing the data loaded
    '''
    pass


def data_preprocess(input_data):
    '''
    input_data : numpy array
                m x n numpy data array
    return : tuple
                (train, test) training data and test data tuple
                each as a numpy array
    '''
    pass

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