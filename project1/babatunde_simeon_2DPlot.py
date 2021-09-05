#!/usr/bin/env python

# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np

# Read in the file, allowing with to handle file clean-ups
fileObject = open("IrisData.txt", "r")
rows = len(fileObject.readlines())
# Reset file pointer to the beginning 
fileObject.seek(0)
ncols = len(fileObject.readline().strip().split("\t"))
fileObject.seek(0)
# Create a 2D array filled with zeros and code the flower type with numbers
data = np.zeros([rows, 5])
for row in range(rows):
    line = fileObject.readline()
    cols = line.strip().split("\t")
    for col in range(ncols - 1):
        data[row,col] = float(cols[col])
    if cols[ncols - 1] == "setosa":
        data[row,ncols-1] = 1
    elif cols[ncols - 1] == "versicolor":
        data[row,ncols-1] = 2
    else:
        data[row,ncols-1] = 3

# Close file object
fileObject.close()

# Plot data with color coding for types setosa=1,versicolor=2 and virginica=3
plt.figure(figsize=(12, 8))
for row in range(rows):
    if(data[row,ncols-1]) == 1:
        plt.scatter(data[row, 0], data[row,2], color="red", marker="o", label="setosa" if row==0 else "")
    elif(data[row,ncols-1]) == 2:
        plt.scatter(data[row, 0], data[row,2], color="blue", marker="v", label="versicolor" if row==50 else "")
    else:
        plt.scatter(data[row, 0], data[row,2], color="green", marker="x", label="virginica" if row==100 else "") 
        
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.title("Plot of Iris Dataset (Sepal Length vs Petal Length)")
plt.legend()
plt.savefig("babatunde_simeon_MyPlot.png", bbox_inches="tight")
plt.show()