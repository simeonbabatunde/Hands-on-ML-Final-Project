from models import BaseModel
import numpy as np

def sigmoid(z):
    '''
    g(z) = 1 / 1 + e^-z
    '''
    return (1 / (1 + np.exp(-z)))

def sigmoid_derivative(g):
    '''
    Give a sigmoid output g, calculate its gradient
    (Gets the slope of that point in the sigmoid)
    '''
    return g * (1 - g)

def cross_entropy_loss(H, Y):
    '''
    Cost(h_w(x), y) = 1/m SUM( -y * ln (h_w(x)) - (1-y)ln(1 - h_w(x)))
    '''
    m = Y.shape[0]
    Y_neg = -1 * Y
    a_term = np.sum(np.multiply(np.log(H), Y_neg))
    b_term = np.sum(np.multiply(np.log(1 - H), (1 - Y)))
    C = (1 / m) * ((a_term) - (b_term))
    return C
    

class NeuralNet(BaseModel.BaseModel):
    '''
    A 3 Layer Neural Network without Regularization and Dropout
    Usage:
        train(train_set): to train the model on the given training set
        test(test_set): to test the learned model. Returns Accuracy Stats
    '''
    def __init__(self):
        '''
        Initializing Model Params and Hyper Params to defaults
        Will be overriden inside the train method
        '''
        self.X = None # training input data. Include X0 = 1
        self.Y = None # Output classes
        self.m = 0
        self.n = 0
        self.no_output = 0
        self.W_0 = None
        self.W_1 = None
        self.no_hidden = 12
        self.alpha = 0.001
        self.no_epochs = 40000
        self.debug = True
        
    def train(self, tr_set):
        
        self.X = tr_set[:, :-1]
        self.Y = tr_set[:, -1:]
        self.m = tr_set.shape[0]
        self.n = self.X.shape[1]
        self.no_hidden = self.n
        self.no_output = self.Y.shape[1]
        
        if self.debug:
            print("===================================")
            print("X " + str(self.X.shape))
            print("Y " + str(self.Y.shape))
            print("M " + str(self.m))
            print("N " + str(self.n))
            print("Y " + str(self.no_output))
        
        # Random init with zero mean
        np.random.seed(1)
        self.W_0 = 2 * np.random.random((self.n, self.no_hidden)) - 1
        self.W_1 = 2 * np.random.random((self.no_hidden, self.no_output)) - 1
        
        for i in range(self.no_epochs):
            '''
            Step 1: Forward Prop
            One step of forward prop across the network for all training examples
            The results of each layer is stored in a(i) i denoting the layer
            a_0 will be the input
            '''                        
            a_0 = self.X
            a_1 = np.dot(a_0, self.W_0)
            a_1 = sigmoid(a_1)
            a_2 = np.dot(a_1, self.W_1)
            a_2 = sigmoid(a_2)
            
            # Output Error
            a_2_err = a_2 - self.Y
            # Calculate Cost Function
            cost = cross_entropy_loss(a_2, self.Y)
            if i % 5000 == 0:
                print("Cost " + str(cost))

            
            # For each point on the sigmoid, we need to calculate the direction of 
            # slope
            a_2_delta = np.multiply(a_2_err, sigmoid_derivative(a_2))
            
            # Hidden Layer Error
            a_1_err = np.dot(a_2_delta, self.W_1.T)
            a_1_delta = np.multiply(a_1_err, sigmoid_derivative(a_1))
            
            self.W_1 -= self.alpha * (np.dot(a_1.T, a_2_delta))
            self.W_0 -= self.alpha * (np.dot(a_0.T, a_1_delta))
            
    def test(ts_set):
        pass

