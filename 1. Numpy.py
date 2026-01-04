# Install NumPy
# Type "pip install numpy" in Terminal or Command Prompt

# Import convention
import numpy as np
print("NumPy version:", np.__version__)                   # Print NumPy version (optional)
"""__________________________________________________________________________________________________________________________________________"""

# Creating an Array from Python list
arr1 = np.array([1, 2, 3, 4, 5])
print("1D Array:", arr1)
"""
[1 2 3 4 5]
"""

# 2D array from nested lists
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print("\n2D Array:\n", arr2)
"""
[[1 2 3]
 [4 5 6]]
"""

# 3D array
arr3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print("\n3D Array:\n", arr3)
"""
[[[1 2] 
[3 4]]
    [[5 6]
    [7 8]]]
"""

print("\n3D Array shape:", arr3.shape)
"""
(2, 2, 2)
"""

print("\nDimensions of arr3:", arr3.ndim)
"""
3
"""

arr4 = np.reshape(arr3, (2, 4))
print("\nReshaped Array (2x4):\n", arr4)
"""
[[1 2 3 4]
[5 6 7 8]]
"""

arr5 = arr3.flatten()
print("\nFlattened Array:", arr5)
"""
[1 2 3 4 5 6 7 8]
"""

arr6 = arr3.T
print("\nTransposed Array:\n", arr6)
"""
[[[1 5]
  [3 7]]

 [[2 6]
  [4 8]]]
"""

arr7= arr3.ravel()                              # Ravel is similar to flatten but returns a view when possible.
print("\nRaveled Array:", arr7)                 # Changes the same array without creating a new copy if possible.
"""                                             # Not possible if the array is not contiguous in memory.
[1 2 3 4 5 6 7 8]                               # Coniguous means that the elements are stored in adjacent memory locations.
"""                                             # Contiguous arrays are more efficient for computations.
"""
Eg of when ravel returns a view:
arr8 = np.array([[1, 2, 3], [4, 5, 6]])
arr9 = arr8.ravel()
arr9[0] = 10
print("\nOriginal array after modifying raveled view:\n", arr8)
"
[[10  2  3]
    [ 4  5  6]]
"
"""

print("\nData type of arr1:", arr1.dtype)
"""
int64
"""

print("\nSize of each element in arr2 (in bytes):", arr2.itemsize)
"""
8
"""

print("\nTotal number of elements in arr2:", arr2.size)
"""
6
"""

print("\nTotal size of arr2 (in bytes):", arr2.nbytes)
"""
48
"""
"""_________________________________________________________________________________________________________________________________________"""

# Zeros array
zeros_arr = np.zeros((3, 4))
print("Zeros array:\n", zeros_arr)
"""
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]
"""

# Ones array
ones_arr = np.ones((2, 3))
print("\nOnes array:\n", ones_arr)
"""
[[1. 1. 1.]
 [1. 1. 1.]]
"""

# Identity matrix
identity = np.eye(3)
print("\nIdentity matrix:\n", identity)
"""
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
"""

# Empty array (uninitialized)
empty_arr = np.empty((2, 2))                       # Note: values are uninitialized and may vary
print("\nEmpty array:\n", empty_arr)               # Creates an array with random, leftover values in memory, not initialized.
"""
[[0. 0.]                  [[6.935e-310 6.935e-310]
 [0. 0.]]       OR        [6.935e-310 6.935e-310]]              Randomly varies
"""

# Full array
full_arr = np.full((2, 3), 7)                     # Allocates an array and sets all elements to the specified value.
print("\nFull array (value=7):\n", full_arr)      # Creates an array and fills every element with the given value.
"""
[[7 7 7]
 [7 7 7]]
"""

"""__________________________________________________________________________________________________________________________________________"""

# arange (like range but returns array)
range_arr = np.arange(0, 10, 2)
print("arange(0, 10, 2):", range_arr)
"""
[0 2 4 6 8]
"""

# linspace (evenly spaced numbers)
lin_arr = np.linspace(0, 1, 5)
print("\nlinspace(0, 1, 5):", lin_arr)
"""
[0.   0.25 0.5  0.75 1.  ]
"""
"""__________________________________________________________________________________________________________________________________________"""

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print("np.negative(a):", np.negative(a))
"""[-1 -2 -3]"""                            # Element-wise negation

print("np.sort(b):", np.sort(b))
"""[4 5 6]"""

print("np.unique([1, 2, 2, 3, 3, 3]):", np.unique([1, 2, 2, 3, 3, 3]))
"""[1 2 3]"""

print("a + b:", a + b)                      # Alternatively, np.add(a, b)
"""[5 7 9]"""                               # Addition of corresponding elements

print("a - b:", a - b)                      # Alternatively, np.subtract(a, b)
"""[-3 -3 -3]"""                            # Subtraction of corresponding elements

print("a * b:", a * b)                      # Alternatively, np.multiply(a, b)
"""[ 4 10 18]"""                            # Multiplication of corresponding elements

print("a / b:", a / b)                      # Alternatively, np.divide(a, b)
"""[0.25 0.4  0.5 ]"""                      # Division of corresponding elements

print("a ** 2:", a ** 2)                    # Alternatively, np.power(a, 2)
"""[1 4 9]"""                               # Element-wise exponentiation

print("a % 2:", a % 2)                      # Alternatively, np.mod(a, 2)
"""[1 0 1]"""                               # Element-wise modulus

print("np.floor_divide(a, b):", np.floor_divide(a, b))
"""[0 0 0]"""                                                             # Floor division of corresponding elements

print("np.absolute([-1, -2, 3]):", np.absolute([-1, -2, 3]))
"""[1 2 3]"""                                                             # Element-wise absolute value

print("np.reciprocal(b.astype(float)):", np.reciprocal(b.astype(float)))
"""[0.25 0.2  0.16666667]"""                                              # Element-wise reciprocal

print("np.dot(a, b):", np.dot(a, b))                                      # Dot product of two arrays
"""32"""                                                                  # Calculated as 1*4 + 2*5 + 3*6

print("np.cross(a, b):", np.cross(a, b))                                  # Cross product of two 3D vectors
"""[-3  6 -3]"""                                                          # Calculated as [2*6 - 3*5, 3*4 - 1*6, 1*5 - 2*4]

print("np.sqrt(a):", np.sqrt(a))                                          # Element-wise square root
"""[1.         1.41421356 1.73205081]"""

print("np.sum(a):", np.sum(a))                                            # Sum of all elements
"""6"""

print("np.prod(a):", np.prod(a))                                          # Product of all elements
"""6"""

print("np.cumsum(a):", np.cumsum(a))                                      # Cumulative sum
"""[1 3 6]"""

print("np.cumprod(a):", np.cumprod(a))                                    # Cumulative product
"""[ 1  2  6]"""
"""__________________________________________________________________________________________________________________________________________"""

print("np.mean(a):", np.mean(a))                                          # Mean (average) of elements
"""2.0"""

print("np.median(a):", np.median(a))                                      # Median
"""2.0"""

print("np.std(a):", np.std(a))                                            # Standard deviation of elements
"""0.816496580927726"""                                                   # sqrt(((1-2)^2 + (2-2)^2 + (3-2)^2) / 3)

print("np.var(a):", np.var(a))                                            # Variance
"""0.6666666666666666"""                                                  # ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3


print("np.any(a > 2):", np.any(a > 2))                                    # Any element satisfies condition
"""True"""

print("np.all(a > 0):", np.all(a > 0))                                    # All elements satisfy condition
"""True"""

print("np.percentile(a, 75):", np.percentile(a, 75))                      # Percentile
"""2.5"""                                                                 # 75th percentile value
"""__________________________________________________________________________________________________________________________________________"""

print("np.max(a):", np.max(a))                                            # Maximum element
"""3"""

print("np.min(a):", np.min(a))                                            # Minimum element
"""1"""

print("np.argmax(a):", np.argmax(a))                                      # Index of the maximum element
"""2"""

print("np.argmin(a):", np.argmin(a))                                      # Index of the minimum element
"""0"""
"""__________________________________________________________________________________________________________________________________________"""

print("np.exp(a):", np.exp(a))                                # e^x
"""[ 2.71828183  7.3890561  20.08553692]"""

print("np.exp2(a):", np.exp2(a))                              # 2^x
"""[2. 4. 8.]"""

print("np.log(a):", np.log(a))                                # Natural log
"""[0.         0.69314718 1.09861229]"""

print("np.log2(a):", np.log2(a))                              # Base-2 log
"""[0.         1.         1.5849625 ]"""

print("np.log10(a):", np.log10(a))                            # Base-10 log
"""[0.         0.30103    0.47712125]"""
"""__________________________________________________________________________________________________________________________________________"""

angles = np.array([0, np.pi/2, np.pi])

print("np.sin(angles):", np.sin(angles))
"""[0.0000000e+00 1.0000000e+00 1.2246468e-16]"""                     # Note: The last value is effectively zero; it appears due to floating-point precision limitations.

print("np.cos(angles):", np.cos(angles))
"""[ 1.0000000e+00  6.1232340e-17 -1.0000000e+00]"""                  # Note: The second value is effectively zero; it appears due to floating-point precision limitations.

print("np.tan(angles):", np.tan(angles))
"""[ 0.0000000e+00  1.6331239e+16 -1.2246468e-16]"""                  # Note: The second value is extremely large due to the tangent of π/2 being undefined; it appears due to floating-point precision limitations.

print("np.arcsin([0, 0.5, 1]):", np.arcsin([0, 0.5, 1]))              # Inverse sine
"""[0.         0.52359878 1.57079633]"""

print("np.arccos([0, 0.5, 1]):", np.arccos([0, 0.5, 1]))              # Inverse cosine
"""[1.57079633 1.04719755 0.        ]"""

print("np.arctan([0, 1, np.inf]):", np.arctan([0, 1, np.inf]))        # Inverse tangent
"""[0.         0.78539816 1.57079633]"""

print("np.sinh([0, 1, 2]):", np.sinh([0, 1, 2]))                      # Hyperbolic sine
"""[0.         1.17520119 3.62686041]"""                              # Calculated as (sinh(x) = (e^x - e^(-x)) / 2)

print("np.cosh([0, 1, 2]):", np.cosh([0, 1, 2]))                      # Hyperbolic cosine
"""[1.         1.54308063 3.76219569]"""                              # Calculated as (cosh(x) = (e^x + e^(-x)) / 2)

print("np.tanh([0, 1, 2]):", np.tanh([0, 1, 2]))                      # Hyperbolic tangent
"""[0.         0.76159416 0.96402758]"""                              # Calculated as (tanh(x) = sinh(x) / cosh(x))

print("np.arcsinh([0, 1, 2]):", np.arcsinh([0, 1, 2]))
"""[0.         0.88137359 1.44363548]"""                              # Calculated as (arcsinh(x) = ln(x + sqrt(x^2 + 1)))

print("np.arccosh([1, 2, 3]):", np.arccosh([1, 2, 3]))
"""[0.         1.3169579  1.76274717]"""                              # Calculated as (arccosh(x) = ln(x + sqrt(x^2 - 1)))

print("np.arctanh([0, 0.5, 0.9]):", np.arctanh([0, 0.5, 0.9]))
"""[0.         0.54930614 1.47221949]"""                              # Calculated as (arctanh(x) = 0.5 * ln((1 + x) / (1 - x)))
"""__________________________________________________________________________________________________________________________________________"""

a = np.array([1, 2, 3])
b = np.array([2, 2, 2])

print("np.equal(a, b):", np.equal(a, b))                              # Alternative to ==
print("np.not_equal(a, b):", np.not_equal(a, b))                      # Alternative to !=
print("np.greater(a, b):", np.greater(a, b))                          # Alternative to >
print("np.greater_equal(a, b):", np.greater_equal(a, b))              # Alternative to >=
print("np.less(a, b):", np.less(a, b))                                # Alternative to <
print("np.less_equal(a, b):", np.less_equal(a, b))                    # Alternative to <=

""" Results:
np.equal(a, b): [False  True False]
np.not_equal(a, b): [ True False  True]
np.greater(a, b): [False False  True]
np.greater_equal(a, b): [False  True  True]
np.less(a, b): [ True False False]
np.less_equal(a, b): [ True  True False]
"""

