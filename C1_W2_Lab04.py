import numpy as np
import matplotlib.pyplot as plt
from lab_utils_multi import zscore_normalize_features, run_gradient_descent_feng
np.set_printoptions(precision=2)

# create target data
x = np.arange(0, 20, 1)
y = 1 + x**2

# reshape X to 2_D matrix
X = x.reshape(-1, 1)

# gradient descent with raw feature
model_w, model_b = run_gradient_descent_feng(X, y, iterations=1000, alpha=1e-2)

# configure and display plot
plt.scatter(x, y, marker="x", c="r", label="Actual Value")
plt.title("no feature engineering")
plt.plot(x, X@model_w + model_b, label="Predicted Value")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# create target data
x = np.arange(0, 20, 1)
y = 1 + x**2

# engineer features
X = x**2

# reshape X to 2_D matrix
X = X.reshape(-1, 1)

# gradient descent with engineered feature
model_w, model_b, = run_gradient_descent_feng(X, y, iterations=10000, alpha=1e-5)

# configure and display plot
plt.scatter(x, y, marker="x", c="r", label="Actual Value")
plt.title("added x**2 feature")
plt.plot(x, np.dot(X, model_w) + model_b, label="Predicted Value")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# create target data
x = np.arange(0, 20, 1)
y = x**2

# engineer features
X = np.c_[x, x**2, x**3]

# gradient descent with engineered features
model_w, model_b = run_gradient_descent_feng(X, y, iterations=10000, alpha=1e-7)

# configure and display plot
plt.scatter(x, y, marker="x", c="r", label="Actual Value")
plt.title("x, x**2, x**3 features")
plt.plot(x, X@model_w + model_b, label="Predicted Value")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# create target data
x = np.arange(0, 20, 1)

# engineer features
X = np.c_[x, x**2, x**3]
X_features = ["x", "x^2", "x^3"]

# plot and display plot
fig, ax = plt.subplots(1, 3, figsize=(12, 3), sharey=True)

for i in range(len(ax)):
    ax[i].scatter(X[:,i], y)
    ax[i].set_xlabel(X_features[i])

ax[0].set_ylabel("y")
plt.show()
