#!/usr/bin/env python

# Author: Simeon Babatunde
# Date: September 16, 2019
# Description: This script contains the implementation of a Binary Classifier using K-Nearest Neighbor KNN approach
# The first part of the code include determining the k(no of nearest neighbor) that yeilds hihest accuracy 

# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import math

nrows=0;

####################################### Useful Functions ########################################
# This function Normalizes the dataset using (x-xmin)/(xmax - xmin) and returns normalized, min and max
def normalize(array):
    normalized = array
    minn = array.min(axis=0)
    maxx = array.max(axis=0)
    rows = (np.size(array, 0))
    for i in range(rows):
        normalized[i][0] = (array[i][0] - minn[0])/(maxx[0] - minn[0])
        normalized[i][1] = (array[i][1] - minn[1])/(maxx[1] - minn[1])
    return normalized, minn, maxx

def euclidean_distance(f1,f2,p1,p2):
    distance = math.sqrt((f1-p1)**2 + (f2-p2)**2)
    return distance

# Compute Distance of all points to a point. Accepts k-Fold array and new point(). Returns array of sorted distance and class
def compute_distance_to_point(fold_array, x1, x2): 
    result = np.zeros([len(fold_array),2])
    for n in range(len(fold_array)):
        result[n][0] = euclidean_distance(fold_array[n][0], fold_array[n][1], x1, x2)
        result[n][1] = fold_array[n][2]
    
    #Sort the result based on distance from the point, idx 0
    result = result[result[:,0].argsort()]
    return result


# Return array of predictions for test data
def predict(train_data, test_data, kv):
    pred_output = np.zeros(len(test_data))
    for p in range(len(test_data)):
        sorted_result = compute_distance_to_point(train_data, test_data[p][0], test_data[p][1])
        sum=0.0
        for t in range(kv):
            sum+=sorted_result[t][1]
        prediction = 1.0 if(sum/kv)>=0.5 else 0.0
        pred_output[p] = prediction
    return pred_output


# This compute error between the prediction and test labels
def compute_error(prediction_array, test_labels):
    miss = np.count_nonzero(prediction_array!=test_labels)
    return miss

# Plot data with color coding for types setosa=1,versicolor=2 and virginica=3
def plot(dataframe):
    plt.figure(figsize=(10, 8))
    for row in range(len(dataframe)):
        if(dataframe[row,2]) == 1.0:
            plt.scatter(dataframe[row, 0], dataframe[row,1], color="green", marker="o", label="TigerFish1" if row==0 else "")
        else:
            plt.scatter(dataframe[row, 0], dataframe[row,1], color="red", marker="x", label="TigerFish0" if row==151 else "") 
            
    plt.xlabel("Body Length (cm)")
    plt.ylabel("Dorsal Fin Length (cm)")
    plt.title("Plot of TigerFish Body Length vs Dorsal Fin Length")
    plt.legend()
    plt.savefig("DataPlot.png", bbox_inches="tight")
    plt.show()
        


################## Function Call Starts Here ###################################

# Prompt user for filename containing dataset and read it in
fileName = input("Enter the name of the file containing the training data: ")

#Confirm that a file name is supplied
if not fileName:
    print("File name is required")
    exit(0)

# Read in the training data file
try:
    with open(fileName, "r") as fileObject:
        nrows = int(fileObject.readline())
        #Create a 3D array filled with zeros
        data = np.zeros([nrows, 3])
        for row in range(nrows):
            line = fileObject.readline()
            cols = line.strip().split("\t")
            for col in range(len(cols)):
                data[row, col] = float(cols[col])
except:
    print("Cannot read file. Supply correct filename")
    exit(0)
    

# Randomly shuffle the array before splitting into train and test
#np.random.seed(500)
#np.random.shuffle(data)
    
# Plot the input dataset
#raw_data = np.copy(data)
#plot(raw_data)

# standardize the dataset
norm_data, mini, maxi = normalize(data)


#Split data into training 80% and test 20%
#train, test = norm_data[:int(nrows*.8), :], norm_data[int(nrows*.8):, :]


########################### K-Fold Corss Validation Starts Here ################
## Initialize no of NN k
#k_nn = 11
# Create 5 fold cross validation on the training data
#splitted = np.split(train, 5)

#for i in range(len(splitted)):
##    print("Fold ", i)
#    fold_test = splitted[i]
#    fold_train = np.zeros(shape=(1, 3)) 
#    for j in range(len(splitted)):
#        if not np.array_equal(fold_test, splitted[j]):
#            fold_train = np.concatenate((fold_train, splitted[j]))
#    fold_train = np.delete(fold_train, 0, 0)
#  
##    plot(fold_train)
#    l=0
#    summary = np.zeros([2,k_nn])
#    for kk in range(1,k_nn*2,2):
#        final_predict = predict(fold_train, fold_test, kk)
#        error = compute_error(final_predict, fold_test[:, 2])
#        summary[0][l] = kk
#        summary[1][l] = error
#        l+=1
#        

######################## Run the Actual Test Data with k=11 ####################
#test_prediction = predict(train, test, 11)
#labels = test[:, 2]
#
#print("Predictions\n",test_prediction)
#print("Labels\n",labels)
#
## Comput the error
#err = compute_error(test_prediction, labels)
#accuracy = (1-(err/len(labels)))
#
#print("Total: ", len(labels), "\tError: ", err, "\tAccuracy: ", accuracy)
#print(mini, "\t", maxi)


while(1):
    try: 
        body_length = float(input("Enter body length in cm: "))
        fin_length = float(input("Enter dorsal-fin length in cm: "))
    except:
        print("Wrong input. Numbers expected and not string. Try again!!!")
        continue
        
    if(body_length==0 and fin_length==0):
        print("Goodbye!!!")
        break
    
    # Normalize input data
    norm_body = (body_length - mini[0])/(maxi[0] - mini[0]) 
    norm_fin = (fin_length - mini[1])/(maxi[1] - mini[1]) 
    # Create a 2d numpy array to wrap user data
    user_input = np.zeros([1,2])
    user_input[0][0] = norm_body
    user_input[0][1] = norm_fin
    # Get Prediction for this particular fish
    fish_class = predict(norm_data, user_input, 11)
    
    print("TigerFish1") if fish_class[0]==1.0 else print("TigerFish0") 