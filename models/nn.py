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
    Cost(h_w(x), y) = 1/m SUM SUMk=0toK(-y*ln(h_w(x)))
    '''
    m = Y.shape[0]
    Y_neg = -1 * Y
    a_term = np.sum(np.multiply(np.log(H), Y_neg))
    b_term = np.sum(np.multiply(np.log(1 - H), (1 - Y)))
    C = (1 / m) * ((a_term) - (b_term))
    return C

def cross_entropy_loss_multiclass(H, Y):
    '''
    Cost(h_w(x), y) = 1/m SUM( -y * ln (h_w(x)) - (1-y)ln(1 - h_w(x)))
    '''
    m = Y.shape[0]
    Y_neg = -1 * Y
    a_term = np.sum(np.log(H) * Y_neg)
    C = (1 / m) * (a_term)
    return C
     
def softmax(z):
    '''
    Squashes the output of the neurons in a layer into output probabilities
    g(z) = (e^z) / ∑_j=0_no_output e^zj.
    '''
    z = z - np.max(z, axis=1, keepdims=True)
    z_exp = np.exp(z)
    z_class_sum = np.sum(z_exp, axis=1, keepdims=True)
    return z_exp / z_class_sum

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
        self.alpha = 0.0001
        #self.no_epochs = 40000
        self.no_epochs = 1000
        self.debug = True
        
    def train(self, tr_set):
        
        self.X = tr_set[:, :-1]
        self.Y = tr_set[:, -1:]
        self.m = tr_set.shape[0]
        # Adding bias x0
        bias = np.zeros((self.m, 1))
        bias += 1
        self.X = np.column_stack((bias, self.X))
        self.n = self.X.shape[1]
        self.no_hidden = self.n
        self.no_output = self.Y.shape[1]
        
        if self.debug:
            print("===================================")
            print("X " + str(self.X.shape))
            print("Y " + str(self.Y.shape))
            print("M " + str(self.m))
            print("N " + str(self.n))
        
        
        # Hardcoded for now
        # TODO: @richard. Try to set it as a param
        self.no_output = 3
        
        # One Hot Encoding
        # https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy/42874726
        targets = self.Y.reshape(-1)
        one_hot_targets = np.eye(self.no_output)[targets.astype('int32')]
        self.Y = one_hot_targets
        
        if self.debug:
            print(targets[85:])
            print(one_hot_targets.shape)
            print(one_hot_targets[85:])
        
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
            a_2 = softmax(a_2)
            
            #print(a_2.shape)
            #print(self.Y.shape)
            # Output Error
            a_2_err = a_2 - self.Y
            # Calculate Cost Function
            cost = cross_entropy_loss_multiclass(a_2, self.Y)
            if i == 0:
                print("Init Cost " + str(cost))
            if i % 50 == 0:
                print("Cost " + str(cost))

            
            # For each point on the sigmoid, we need to calculate the direction of 
            # slope
            #a_2_delta = np.multiply(a_2_err, sigmoid_derivative(a_2))
            # No more sigmoids in this layer
            a_2_delta = a_2_err
            
            # Hidden Layer Error
            a_1_err = np.dot(a_2_delta, self.W_1.T)
            a_1_delta = np.multiply(a_1_err, sigmoid_derivative(a_1))
            
            self.W_1 -= self.alpha * (np.dot(a_1.T, a_2_delta))
            self.W_0 -= self.alpha * (np.dot(a_0.T, a_1_delta))
            
    def test(ts_set):
        pass

