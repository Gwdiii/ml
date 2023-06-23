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

# configure and display
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

# plot and display
fig, ax = plt.subplots(1, 3, figsize=(12, 3), sharey=True)

for i in range(len(ax)):
    ax[i].scatter(X[:,i], y)
    ax[i].set_xlabel(X_features[i])

ax[0].set_ylabel("y")
plt.show()

# create target data
x = np.arange(0, 20, 1)
X = np.c_[x, x**2, x**2]
print(f"Peak to Peak range by column in Raw")

# add mean_normalization
X = zscore_normalize_features(X)
print(f"Peak to Peak range by column in Normalized X: {np.ptp(X, axis=0)}")

# add more aggresive alpha value
# create target data
x = np.arange(0, 20, 1)
y = x**2

# add mean_normalization
X = np.c_[x, x**2, x**3]
X = zscore_normalize_features(X)

# gradient descent with scaled features, high alpha
model_w, model_b = run_gradient_descent_feng(X, y, iterations=100000, alpha=1e-1)

plt.scatter(x, y, marker="x", c="r", label="Actual Value")
plt.title("Normalized x, x**2, x**3 features")
plt.plot(x, X@model_w + model_b, label="Predicted Value")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# complex function
# create target data
x = np.arange(0, 20, 1)
y = np.cos(x/2)

# add engineered features
X = np.c_[x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13]
X = zscore_normalize_features(X)

# gradient descent
model_w, model_b = run_gradient_descent_feng(X, y, iterations=1000000, alpha=1e-1)

# plot and display
plt.scatter(x, y, marker="x", c="r", label="Actual Value")
plt.title("Normalized x**j features")
plt.plot(x, X@model_w + model_b, label="Predicted Value")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
