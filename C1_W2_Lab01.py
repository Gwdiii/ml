import numpy as np
import time

a = np.zeros(4);                print(f'np.zeros(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}')
a = np.zeros((4,));             print(f'np.zeros(4,): a = {a}, a shape = {a.shape}, a data type = {a.dtype}')
a = np.random.random_sample(4); print(f'np.random.random_sample(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}')

a = np.arange(4.);              print(f'np.arange(4.): a = {a}, a shape = {a.shape}, a data type = {a.dtype}')
a = np.random.rand(4);          print(f'np.random.rand(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}')

a = np.array([5, 4, 3, 2]);     print(f'np.array([5, 4, 3, 2]): a = {a}, a shape = {a.shape}, a data type = {a.dtype}')
a = np.array([5., 4, 3, 2]);    print(f'np.array([5, 4, 3, 2]): a = {a}, a shape = {a.shape}, a data type = {a.dtype}')

# vector indexing operations on 1-D vectors
a = np.arange(10)
print(a)

# access an element
print(f'a[2].shape: {a[2].shape} a[2] = {a[2]}, Accessing an element returns a scalar')

# access the last element, negative indices count from the end
print(f'a[-1] = {a[-1]}')

# indices must be within the range of the vector or they will produce an error
try:
    c = a[10]
except Exception as e:
    print("The error message you'll see is:")
    print(e)

# vector slicing operations
a = np.arange(10); print(f"a = {a}")

# access 5 consecutive elements (start: stop: step)
c = a[2:7:1];      print(f"a[2:7:1] = {c}")

# access every other element from start to stop
c = a[2:7:2];      print(f"a[2:7:2] = {c}")

# access all elements from index 3 forward
c = a[3:];          print(f"a[3:] = {c}")

# access all elements until index 3
c = a[:3];          print(f"a[:3] = {c}")

# access all elements
c = a[:];           print(f"a[:] = {c}")


a = np.array([1, 2, 3, 4])
print(f"a: {a}")

# negates elements of a
b = -a
print(f"b = -a: {b}")

# sum all elements of a, returns a scalar
b = np.sum(a)
print(f"b = np.sum(a): {b}")

# Average all elements of a
b = np.mean(a)
print(f"b = np.mean(a): {b}")

# Square all elements of a
b = a ** 2
print(f"b = a ** 2: {b}")

a = np.array([1, 2, 3, 4])
b = np.array([-1, -2, 3, 4])
print(f"Binary operators work element wise: {a + b}")

# Try a mismatched vector operation
c = np.array([1, 2])

try:
    d = a + c
except Exception as e:
    print("The error message you'll see is:")
    print(e)

a = np.array([1, 2, 3, 4])

# Mutiply a by a scalar
b = 5 * a
print(f"b = 5 * a: {b}")

# Compute the dot product of two vectors
#
# Args:
# a (ndarray (n,)): input vector
# b (ndarray (n,)): input vector with same dimension as a
#
# Returns:
# x (scalar)
def my_dot(a, b):
    x = 0
    for i in range(a.shape[0]): x = x + a[i] * b[i]
    return x

# test 1-D
a = np.array([ 1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])
print(f"my_dot(a, b) = {my_dot(a, b)}")

a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])
c = np.dot(a, b)
print(f"NumPy 1-D np.dot(a, b) = {c}, np.dot(a, b).shape = {c.shape}")

c = np.dot(b, a)
print(f"NumPy 1-D np.dot(a, b) = {c}, np.dot(a, b).shape = {c.shape}")

np.random.seed(1)
a = np.random.rand(10000000)
b = np.random.rand(10000000)

# capture start time
tic = time.time()
c = np.dot(a, b)
# capture end time
toc = time.time()

print(f"np.dot(a, b) - {c:.4f}")
print(f"Vectorized version duration: {1000 * (toc-tic) : .4f} ms")

# capture start time
tic = time.time()
c = my_dot(a, b)
# capture end time
toc = time.time()

print(f"my_dot(a, b) - {c:.4f}")
print(f"Loop version duration: {1000 * (toc-tic) : .4f} ms")

del(a)
del(b)

# show common course 1 example
X = np.array([[1], [2], [3], [4]])
w = np.array([2])
c = np.dot(X[1], w)

print(f"X[1] has shape {X[1].shape}")
print(f"w has shape {w.shape}")
print(f"c has shape {c.shape}")

# Matrices /////////////////////////////////////////////////////

a = np.zeros((1, 5))
print(f"a shape = {a.shape}, a = {a}")

a = np.zeros((2, 1))
print(f"a shape = {a.shape}, a = {a}")

a = np.random.random_sample((1, 1))
print(f"a shape = {a.shape}, a = {a}")

# NumPy routines which allocate memory and
# fill with user specified values
a = np.array([[5], [4], [3]])
print(f" a shape = {a.shape}, np.array: a = {a}")

# vector indexing operations on matrices
a = np.arange(6).reshape(-1, 2)
print(f"a.shape: {a.shape}, a = {a}")

# access an element
print(f"a[2,0].shape: {a[2, 0].shape}, a[2,0], = {a[2, 0]}, type(a[2, 0]) = {type(a[2, 0])}")

# access a row
print(f"a[2].shape: {a[2].shape}, a[2] = {a[2]}, type(a[2]) = {type(a[2])}")

# vector 2-D slicing operations
a = np.arange(20).reshape(-1, 10)
print(f"a = {a}")
# access 5 consecutive elements (start: stop: step)
print(f"a[0, 2:7:1] = {a[0, 2:7:1]}, a[0, 2:7:1].shape = {a[0, 2:7:1].shape}, a 1-D array")
# access 5 consecutive elements (start: stop: step) in two rows
print(f"a[:, 2:7:1] = {a[:, 2:7:1]}, a[:, 2:7:1].shape = {a[:, 2:7:1].shape}, a 2-D array")
# access all elements
print(f"a[:, :]= {a[:, :]}, a[:, :].shape = {a[:, :].shape}")
# access all elements in one row (very common usage)
print(f"a[1, :] = {a[1, :]}, a[1, :].shape = {a[1, :].shape}, a 1-D array")
# same as
print(f"a[1] = {a[1]}, a[1].shape = {a[1].shape}, a 1-D array")
