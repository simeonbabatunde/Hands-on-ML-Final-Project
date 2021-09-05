#!/usr/bin/env python

# Author: Simeon Babatunde
# Date: September 24, 2019
# Description: This script contains the implementation of a regression hypothesis function that predict a 
# student’s grade point average based on how many minutes per week they study and how many ounces of beer 
# they consume per week. The polynomial regression model is based on (y = w0 + w1x1 + w2x2 + w3x1x2 + w4x12 + w5x22) 

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

def h_x(w0, w1, w2, w3, w4, w5, x1, x2):
    '''
    This is the polynomial hypothesis function used by the ML algorithm to predict student's grade
    based on minutes studying per week and ounces of beer per week.
    Args: Features and weights
    Returns: prediction output (Grade point)
    '''
    return (w0 + w1*x1 + w2*x2 + w3*x1*x2 + w4*(x1**2) + w5*(x2**2))


def J_w(w0, w1, w2, w3, w4, w5, training_data): 
    '''
    This is the squred error cost function which computes difference between output and prediction
    Args: Weights and 3-column training data (x1, x2, y) 
    Retrurns error cost
    '''
    num_rows = len(training_data)
    sum = 0.0
    
    for i in range(num_rows):
        sum += (h_x(w0, w1, w2, w3, w4, w5, training_data[i][0], training_data[i][1]) - training_data[i][2])**2
    
    return sum/(2*num_rows)

def partial_derivatives(wt0, wt1, wt2, wt3, wt4, wt5, train_data):
    '''
    This computes the pertial derivatives of the cost function w.r.t the w's
    Args: Weights and 3-column training data (x1, x2, y)
    Returns: Array of size 6, containing partial derivative values for w_0 to w_5
    '''
    derivative_array = np.zeros(6);
    sum0=sum1=sum2=sum3=sum4=sum5=0.0
    rows = len(train_data)
    
    for j in range(rows):
        sum0 += (h_x(wt0, wt1, wt2, wt3, wt4, wt5, train_data[j][0], train_data[j][1]) - train_data[j][2])
        sum1 += ((h_x(wt0, wt1, wt2, wt3, wt4, wt5, train_data[j][0], train_data[j][1]) - train_data[j][2]) * train_data[j][0])
        sum2 += ((h_x(wt0, wt1, wt2, wt3, wt4, wt5, train_data[j][0], train_data[j][1]) - train_data[j][2]) * train_data[j][1])
        sum3 += ((h_x(wt0, wt1, wt2, wt3, wt4, wt5, train_data[j][0], train_data[j][1]) - train_data[j][2]) * (train_data[j][0]*train_data[j][1]))
        sum4 += ((h_x(wt0, wt1, wt2, wt3, wt4, wt5, train_data[j][0], train_data[j][1]) - train_data[j][2]) * (train_data[j][0]**2))
        sum5 += ((h_x(wt0, wt1, wt2, wt3, wt4, wt5, train_data[j][0], train_data[j][1]) - train_data[j][2]) * (train_data[j][1]**2))
    
    derivative_array[0] = sum0/rows
    derivative_array[1] = sum1/rows
    derivative_array[2] = sum2/rows
    derivative_array[3] = sum3/rows
    derivative_array[4] = sum4/rows
    derivative_array[5] = sum5/rows
    
    return derivative_array


def gradient_descent(w_0, w_1, w_2, w_3, w_4, w_5, alpha, tr_data):
    '''
    This function computes the gradient descent i.e. new values of w's to minimize the 
    cost function.
    Args: Initial weigths, learning rate (alpha) and the train data
    Returns: New weights w0-to-w5
    '''
    derivatives = partial_derivatives(w_0, w_1, w_2, w_3, w_4, w_5, tr_data)
    #print(derivatives)
    
    temp0 = w_0 - alpha * derivatives[0]
    temp1 = w_1 - alpha * derivatives[1]
    temp2 = w_2 - alpha * derivatives[2]
    temp3 = w_3 - alpha * derivatives[3]
    temp4 = w_4 - alpha * derivatives[4]
    temp5 = w_5 - alpha * derivatives[5]
    
    return temp0, temp1, temp2, temp3, temp4, temp5
    
    

# Return array of predictions for test data
def predict(wtt0, wtt1, wtt2, wtt3, wtt4, wtt5, test_data):
    '''
    This function takes in features and generate presictions 
    Args: Array of input features
    Retrurns: Numpy array of predictions
    '''
    rows = len(test_data)
    pred_output = np.zeros(rows)
    for k in range(rows):
        pred_output[k] = h_x(wtt0, wtt1, wtt2, wtt3, wtt4, wtt5, test_data[k][0], test_data[k][1])
        
    return pred_output


## This compute error between the prediction and test labels
#def compute_error(prediction_array, test_labels):
#    miss = np.count_nonzero(prediction_array!=test_labels)
#    return miss
#
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
def plot_3d(dataframe):
    from mpl_toolkits.mplot3d import axis3d
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(dataframe[:,0], dataframe[:,1], dataframe[:,2], c='b', marker='o')
    
    ax.set_xlabel("Minutes Spent Studying per Week")
    ax.set_ylabel("Ounces of Beer Consumed per Week")
    ax.set_zlabel("Semester GPA")
            
    plt.title("Plot of Initial Dataset")
    plt.savefig("InitialData.png", bbox_inches="tight")
    plt.show()
        


################## Function Call Starts Here ###################################

## Initialize file name
#fileName = "GPAData.txt"
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
##plot_3d(data)
#
#
## Scale the dataset using standardization
#norm_data, mean, std = standardize(data)
#
#
##Split data into training 70% and test 30%
##train, test = data[:int(nrows*.7), :], data[int(nrows*.7):, :]
#train, test = norm_data[:int(nrows*.7), :], norm_data[int(nrows*.7):, :]
#
############################ Training starts here ###############################
## Initialize the weights and learning rate alpha.
#w0, w1, w2, w3, w4, w5 = 50, 72, 48, 5, 12, 10
#alpha = 0.01
#epocs = 3000
#error_epocs = np.zeros([epocs,2])
#
#j_error = J_w(w0, w1, w2, w3, w4, w5, train)
#error_epocs[0][0] = 0
#error_epocs[0][1] = j_error
#
#
#for row in range(epocs-1):
#    w0, w1, w2, w3, w4, w5 = gradient_descent(w0, w1, w2, w3, w4, w5, alpha, train)
#    j_error = J_w(w0, w1, w2, w3, w4, w5, train)
#    error_epocs[row+1][0] = row+1
#    error_epocs[row+1][1] = j_error
#
##plot_error(error_epocs)
##print(w0, w1, w2, w3, w4, w5)
#print(error_epocs[-1, 1])

######################### Run Test Data on hypothesis function ##################
##test_prediction = predict(w0, w1, w2, w3, w4, w5, test)
##np.around(test_prediction, 2, test_prediction)
##labels = test[:, 2]
#
## Compute Error J on test data
#test_error = J_w(w0, w1, w2, w3, w4, w5, test)
##print("\n",test_error)
#test_J =  0.00015148778367610205


w0 = 1.204926783686873
w1 = 1.1689073231292413
w2 = -0.0006834056528370415
w3 = -0.0014895007193559378
w4 = 0.27818631239172503
w5 = -0.012474841502375435

mean_study, mean_beer, std_study, std_beer = 541.3, 153.0, 270.10925567, 76.96583658


while(1):
    try: 
        study_time = float(input("Enter minutes spent studying per week: "))
        beer_ounces = float(input("Enter ounces of beer consumed per week: "))
    except:
        print("Wrong input. Numbers expected. Try again!!!")
        continue
        
    if(study_time==0 and beer_ounces==0):
        print("Goodbye!!!")
        break
    
    # Normalize input data
    scaled_study_time = (study_time - mean_study)/std_study
    scaled_beer_ounces = (beer_ounces - mean_beer)/std_beer  
    # Create a 2d numpy array to wrap user data
    user_input = np.zeros([1,2])
    user_input[0][0] = scaled_study_time
    user_input[0][1] = scaled_beer_ounces
    # Get GPA Prediction for student's input
    gpa = predict(w0, w1, w2, w3, w4, w5, user_input)
    
    if gpa[0] < 0:
        print("{:.2f}".format(abs(gpa[0])))
    elif gpa[0] > 4.0:
        print(4.0)
    else:
        print("{:.2f}".format(gpa[0]))