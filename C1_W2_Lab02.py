import copy, math
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('./deeplearning.mplstyle')
np.set_printoptions(precision = 2)

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

# data is stored in numpy array/matrix
print(f"X Shape: {X_train.shape}, X Type: {type(X_train)}")
print (X_train)
print(f"y Shape: {y_train.shape}, y Type: {type(y_train)}")
print (y_train)

b_init = 785.1811367994083
w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")

# single prediction using linear regression
# Args:
#     x (ndarray): Shape (n,) example with multiple features
#     w (ndarray): Shape (n,) model parameters
#     b  (scalar): model parameter
# Returns:
#     p (scalar): prediction
def predict_single_loop(x, w, b):
    n = x.shape[0]
    p = 0

    for i in range(n):
        p_i = x[i] * w[i]
        p = p + p_i

    p = p + b
    return p

# get a row from out training data
x_vec = X_train[0,:]
print(f"x_vec shape: {x_vec.shape}, x_vec value: {x_vec}")
#make a prediction
f_wb = predict_single_loop(x_vec, w_init, b_init)
print(f"f_wb shape: {f_wb.shape}, prediction: {f_wb}")

# single prediction using linear regression
# Args:
#     x (ndarray): Shape (n,) example with multiple features
#     w (ndarray): Shape (n,) model parameters
#     b  (scalar): model parameter
# Returns:
#     p (scalar): prediction
def predict(x, w, b):
    p = np.dot(x, w) + b
    return p

x_vec = X_train[0,:]
print(f"x_vec shape: {x_vec.shape}, x_vec value: {x_vec}")

f_wb = predict(x_vec, w_init, b_init)
print(f"f_wb shape: {f_wb.shape}, prediction: {f_wb}")

# compute cost
# Args:
#     X (ndarray (m,n)): Data, m examples with n features
#     y (ndarray (m, )): target values
#     w (ndarray (n, )): model parameters
#     b (scalar)       : model parameter
# Returns:
#     cost (scalar): cost
def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0.

    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost = cost + (f_wb_i - y[i]) ** 2

    cost = cost / (2 * m)
    return cost

# Compute and display cost using our pre-chosen optimal parameters
cost = compute_cost(X_train, y_train, w_init, b_init)
print(f"Cost at optimal w: {cost}")

# compute gradient for linear regression
# Args:
#     X (ndarray (m,n)): Data, m examples with n features
#     y (ndarray (m, )): target values
#     w (ndarray (n, )): model parameters
#     b (scalar)       : model parameter
# Returns:
#     dj_dw (ndarray (n, )): gradient of the cost w.r.t. the parameters w
#     dj_db (scalar)       : gradient of the cost w.r.t. the parameter b
def compute_gradient(X, y, w, b):
    # number of examples, number of features
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]

        for j in range(n): dj_dw[j] = dj_dw[j] + err * X[i, j]

        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_db, dj_dw

#Compute and display gradient
tmp_dj_db, tmp_dj_dw = compute_gradient(X_train, y_train, w_init, b_init)
print(f"dj_db at initial w,b: {tmp_dj_db}")
print(f"dj_dw at initial w,b: {tmp_dj_dw}")

    
