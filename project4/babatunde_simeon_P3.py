#!/usr/bin/env python

# Author: Simeon Babatunde
# Date: October 12, 2019
# Description: This script contains the implementation of a logistic regression hypothesis function that classifies a 
# tigerfish as TigerFish0 or TigerFish1 based on the Body lenght and Fin length. The decision boundary is based 
# on w0 + w1x1 + w2x2 + w3x1x2

# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np

nrows=0;

####################################### Useful Functions ########################################
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
    rows = (np.size(array, 0))
    for i in range(rows):
        standardized[i][0] = (array[i][0] - mean[0])/std[0]
        standardized[i][1] = (array[i][1] - mean[1])/std[1]
    return standardized, mean, std

def h_x(w0, w1, w2, w3, x1, x2):
    '''
    This is the polynomial hypothesis function used by the ML algorithm to predict student's grade
    based on minutes studying per week and ounces of beer per week.
    Args: Features and weights
    Returns: prediction output (0 to 1)
    '''
    z = w0 + w1*x1 + w2*x2 + w3*x1*x2
    return 1.0/(1 + np.exp(-z))


def J_w(w0, w1, w2, w3, training_data): 
    '''
    This is the Cost function 
    Args: Weights and 3-column training data (x1, x2, y) 
    Retrurns error cost
    '''
    num_rows = len(training_data)
    sum = 0.0
    
    for i in range(num_rows):
        h_of_x = h_x(w0, w1, w2, w3, training_data[i][0], training_data[i][1])
        sum+= (((training_data[i][2] * np.log(h_of_x)) + ((1.0 - training_data[i][2]) * np.log(1.0 - h_of_x))))
    
    return -sum/num_rows

def partial_derivatives(wt0, wt1, wt2, wt3, train_data):
    '''
    This computes the pertial derivatives of the cost function w.r.t the w's
    Args: Weights and 3-column training data (x1, x2, y)
    Returns: Array of size 6, containing partial derivative values for w_0 to w_5
    '''
    derivative_array = np.zeros(4)
    sum0=sum1=sum2=sum3=0.0
    rows = len(train_data)
    
    for j in range(rows):
        sum0 += (h_x(wt0, wt1, wt2, wt3, train_data[j][0], train_data[j][1]) - train_data[j][2])
        sum1 += ((h_x(wt0, wt1, wt2, wt3, train_data[j][0], train_data[j][1]) - train_data[j][2]) * train_data[j][0])
        sum2 += ((h_x(wt0, wt1, wt2, wt3, train_data[j][0], train_data[j][1]) - train_data[j][2]) * train_data[j][1])
        sum3 += ((h_x(wt0, wt1, wt2, wt3, train_data[j][0], train_data[j][1]) - train_data[j][2]) * (train_data[j][0]*train_data[j][1]))
        
    derivative_array[0] = sum0/rows
    derivative_array[1] = sum1/rows
    derivative_array[2] = sum2/rows
    derivative_array[3] = sum3/rows
    
    
    return derivative_array


def gradient_descent(w_0, w_1, w_2, w_3, alpha, tr_data):
    '''
    This function computes the gradient descent i.e. new values of w's to minimize the 
    cost function.
    Args: Initial weigths, learning rate (alpha) and the train data
    Returns: New weights w0-to-w5
    '''
    derivatives = partial_derivatives(w_0, w_1, w_2, w_3, tr_data)
    #print(derivatives)
    
    temp0 = w_0 - alpha * derivatives[0]
    temp1 = w_1 - alpha * derivatives[1]
    temp2 = w_2 - alpha * derivatives[2]
    temp3 = w_3 - alpha * derivatives[3]
    
    return temp0, temp1, temp2, temp3
    
    

# Return array of predictions for test data
def predict(wtt0, wtt1, wtt2, wtt3, test_data):
    '''
    This function takes in features and generate predictions 
    Args: Array of input features
    Retrurns: Numpy array of predictions 1 0r 0 based on the class
    '''
    rows = len(test_data)
    pred_output = np.zeros(rows)
    for k in range(rows):
        pred_output[k] = 1.0 if(h_x(wtt0, wtt1, wtt2, wtt3, test_data[k][0], test_data[k][1]) >= 0.5) else 0
        
    return pred_output


## This compute error between the prediction and test labels
def compute_error(prediction_array, test_labels):
    miss = np.count_nonzero(prediction_array!=test_labels)
    return miss

# Plot of error against epochs
def plot_error(dataframe):
    plt.figure(figsize=(12, 8))
    for row in range(len(dataframe)):
        plt.scatter(dataframe[row, 0], dataframe[row,1], color="green", marker="o")
            
    plt.xlabel("Epochs")
    plt.ylabel("Error J(w)")
    plt.title("Plot of Epochs against Error J")
    plt.savefig("Error.png", bbox_inches="tight")
    plt.show()

# Plot of error against epochs
def plot_data(dataframe):
    labels = dataframe[:, -1]
    tigerfish1 = dataframe[labels == 1]
    tigerfish0 = dataframe[labels == 0]
    
    # Decision Boundary
#    xrange = [np.min(dataframe[:, 0] - 5), np.max(dataframe[:, 0] + 5)]
#    xval = np.arange(xrange[0], xrange[1], 0.5)
#    yval = np.zeros(len(xval))
#    for n in range(len(xval)):
#        yval[n] = -(w0 + w1*xval[n])/(w2 + w3*xval[n])
    
#    print(xval, yval)
    
    plt.scatter(tigerfish1[:, 0], tigerfish1[:, 1], s=10, label='TigerFish1')
    plt.scatter(tigerfish0[:, 0], tigerfish0[:, 1], s=10, label='TigerFish0')
#    plt.plot(xval, yval, label='D Boundary', color='brown')
    plt.legend()            
    plt.title("Plot of TigerFish Body Length vs Dorsal Fin Length")
    plt.xlabel("Body Length (cm)")
    plt.ylabel("Dorsal Fin Length (cm)")
    plt.savefig("InitialData.png")
    plt.show()
        


################## Function Call Starts Here ###################################

### Initialize file name
#fileName = "FF02.txt"
#
## Read in the training data file
#try:
#    with open(fileName, "r") as fileObject:
#        nrows = int(fileObject.readline())
#        #Create a 3D array filled with zeros
#        data = np.zeros([nrows, 3])
#        for row in range(nrows):
#            line = fileObject.readline()
#            cols = line.strip().split("\t")
#            for col in range(len(cols)):
#                data[row, col] = float(cols[col])
#except:
#    print("Cannot read file. Ensure data file \"DPAData.txt\" is located in this directory")
#    exit(0)
#    
#
## Randomly shuffle the array before splitting into train and test
#np.random.seed(500)
#np.random.shuffle(data)
#
##plot_data(data)
#
##
### Scale the dataset using standardization
#norm_data, mean, std = standardize(data)
#
###Split data into training 70% and test 30%
#train, test = norm_data[:int(nrows*.7), :], norm_data[int(nrows*.7):, :]
#
############################# Training starts here ###############################
### Initialize the weights and learning rate alpha.
#w0, w1, w2, w3 = 0.5, 1.2, 2.8, 0.8
#alpha = 0.05
#epocs = 5000
#error_epocs = np.zeros([epocs,2])
#
#j_error = J_w(w0, w1, w2, w3, train)
#
#error_epocs[0][0] = 0
#error_epocs[0][1] = j_error
#
#for row in range(epocs-1):
#    w0, w1, w2, w3 = gradient_descent(w0, w1, w2, w3, alpha, train)
#    j_error = J_w(w0, w1, w2, w3, train)
#    error_epocs[row+1][0] = row+1
#    error_epocs[row+1][1] = j_error

##plot_error(error_epocs)
#print(error_epocs[0,:])
#print(w0, w1, w2, w3)
##print(mean, std)
#

w0 = -0.32039835972176695
w1 = -2.411093368019604
w2 = -3.3780505239011958
w3 = 1.084723542839459
mean_body, mean_fin = 88.073, 12.47466667
std_body, std_fin = 13.04250632, 3.63902343
########################## Run Test Data on hypothesis function ##################
#test_prediction = predict(w0, w1, w2, w3, test)
#labels = test[:, 2]

# Compute Confusion Matrix
#y_actu = pd.Series(labels, name='Actual')
#y_pred = pd.Series(test_prediction, name='Predicted')
#df_confusion = pd.crosstab(y_actu, y_pred)
#print(df_confusion)

#print(test_prediction, "\n", labels)
### Compute Error on test data
#print(J_w(w0, w1, w2, w3, test))
#error = compute_error(test_prediction, labels)
#print(error, len(test_prediction))
#
#


while(1):
    try: 
        body_length = float(input("Enter body length in cm: "))
        fin_length = float(input("Enter dorsal-fin length in cm: "))
    except:
        print("Wrong input. Numbers expected. Try again!!!")
        continue
        
    if(body_length==0 and fin_length==0):
        print("Goodbye!!!")
        break
    
    # Normalize input data
    scaled_body_length = (body_length - mean_body)/std_body
    scaled_fin_length = (fin_length - mean_fin)/std_fin
        
    # Create a 2d numpy array to wrap user data
    user_input = np.zeros([1,2])
    user_input[0][0] = scaled_body_length
    user_input[0][1] = scaled_fin_length
    # Get Prediction for user input
    fish_class = predict(w0, w1, w2, w3, user_input)
    print("TigerFish1") if fish_class[0]==1.0 else print("TigerFish0")
    
    