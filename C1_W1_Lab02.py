import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')

m = 3
sum = (0.5-1)**2 + (1-2)**2 + (1.5-3)**2
avg = (1 / (2*m)) * sum
print(avg)

#x_train is the input variable (size in 1000 sqft)
#y_train is the target (price in 1000s of dollars)
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

print(f'x_train = {x_train}')
print(f'y_train = {y_train}')

#m is the number of training examples
m = x_train.shape[0]

print(f'x_train.shape: {x_train.shape}')
print(f'Number of training examples is: {m}')

i = 0 #change this to 1 to see (x^(1), y^(1))

x_i = x_train[i]
y_i = y_train[i]
print(f'(x^({i}), y^({i})) = ({x_i}, {y_i})')

#plot the data points
plt.scatter(x_train, y_train, marker='x', c='r')

# set the title
plt.title('Housing Prices')

# set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')

# set the x-axis label
plt.xlabel('Size (in 1000s of sqft)')
plt.show()

w = 200
b = 100
print(f'w: {w}')
print(f'b: {b}')

# Computes the prediction of a linear model
# Args:
#     x (ndarray, (m,)): Data, m examples
#     w, b (scalar)    : model parameters
# Returns:
#     y (ndarray, (m,)): target values
def compute_model_output(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)

    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb

tmp_f_wb = compute_model_output(x_train, w, b,)

#plot the model prediction
plt.plot(x_train, tmp_f_wb, c='b', label='Actual Values')

#plot the data points
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')

#set the title
plt.title('Housing Prices')

#set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')

#set the y-axis label
plt.xlabel('Size (in 1000s of sqft)')
plt.legend()
plt.show()
