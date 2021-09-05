from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def train(self, tr_set):
        '''
        Every model should have a train method that takes in a training data set in the form of 
        a numpy array and trains on it. It sets all of the required parameters as states in the class
        '''
        pass    
    @abstractmethod
    def test(self, ts_set):
        '''
        Every model should have a test method that takes in a test data set in the form of a numpy
        array and executes the model for each data point.
        Returns: Predicted values for each data point in the form of a List
        '''
        pass

