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
