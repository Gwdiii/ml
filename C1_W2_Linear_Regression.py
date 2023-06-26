import numpy             as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math

def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities) 
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0] 
    
    # You need to return this variable correctly
    total_cost = 0
    
    ### START CODE HERE ###
    cost = 0

    for i in range(m):

        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i]) ** 2
    
    total_cost = 1 / (2 * m) * cost
    
    ### END CODE HERE ### 

    return total_cost 

# load the dataset
x_train, y_train = load_data()

# print x_train
print(f"Type of x_train: {type(x_train)}")
print(f"First five elements of x_train are:\n{x_train[:5]}")

# print y_train
print(f"Type of y_train: {type(y_train)}")
print(f"First five elements of y_train are:\n{y_train[:5]}")

# print dimensions of params
print(f"The shape of x_train is: {x_train.shape}")
print(f"The shape of y_train is: {y_train.shape}")
print(f"Number of training examples (m) is: {len(x_train)}")

# plot data
plt.scatter(x_train, y_train, marker="x", c="r")
# set title
plt.title("Profits vs. Population per city")
# set y-axis label
plt.ylabel("Profit in $10,000s")
# set x-axis label
plt.xlabel("Population of City in 10,000s")
# display
plt.show()

# Compute and display gradient with w initialized to zeroes
initial_w = 0
initial_b = 0

tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, initial_w, initial_b)
print('Gradient at initial w, b (zeros):', tmp_dj_dw, tmp_dj_db)

