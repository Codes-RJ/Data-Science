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

print("\nSwap axes (0,1):\n", np.swapaxes(arr, 0, 1))
"""
[[1 5]
 [2 6]
 [3 7]
 [4 8]]
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

# np.copy() - Return array copy
print("\nArray copy of arr1:", np.copy(arr1))
"""[1 2 3 4 5]"""

# np.repeat() - Repeat elements
print("\nArray with elements repeated twice:", np.repeat(arr1, 2))
"""[1 1 2 2 3 3 4 4 5 5]"""

# np.tile() - Construct array by repeating
print("\nArray tiled twice:", np.tile(arr1, 2))
"""[1 2 3 4 5 1 2 3 4 5]"""
"""_________________________________________________________________________________________________________________________________________"""

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print("\nElement at [1, 2]:", arr[1, 2])
"""6"""

print("First row:", arr[0])
"""[1 2 3]"""

print("Last column:", arr[:, -1])
"""[3 6 9]"""

print("Middle column:", arr[:, 1])
"""[2 5 8]"""

print("Sub-array (rows 0-1, cols 1-2):\n", arr[0:2, 1:3])
"""
[[2 3]
  [5 6]]
"""

print("Every second element in flattened array:", arr.flatten()[::2])
"""[1 3 4 6 7 9]"""

print("Reversed array:\n", arr[::-1, ::-1])
"""
[[9 8 7]
  [6 5 4]
  [3 2 1]]
"""

print("Elements > 3:", arr[arr > 3])
"""[4 5 6 7 8 9]"""

print("Even elements:", arr[arr % 2 == 0])
"""[2 4 6 8]"""

indices = [0, 2, 4]

print("Elements at indices [0, 2, 4]:", arr[indices])
"""[1 3 5]"""
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

print("np.square(a):", np.square(a))                                      # Element-wise square
"""[1 4 9]"""

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

print("np.diff(a):", np.diff(a))                                          # Discrete difference
"""[1 1]"""                                                               # Computes the difference between consecutive elements
"""__________________________________________________________________________________________________________________________________________"""

arr = np.array([1.234, 2.5, 3.5])

print("np.round(arr, 2):", np.round(arr, 2))               # Rounds to 2 decimal places
"""[1.23 2.5  3.5 ]"""

print("np.floor(arr):", np.floor(arr))                     # Rounds down to nearest integer (lesser)
"""[1. 2. 3.]"""

print("np.ceil(arr):", np.ceil(arr))                       # Rounds up to nearest integer (greater)
"""[2. 3. 4.]"""

print("np.trunc(arr):", np.trunc(arr))                     # Removes decimal part
"""[1. 2. 3.]"""

print("np.fix(arr):", np.fix(arr))                         # Rounds towards zero
"""[1. 2. 3.]"""

print("np.modf(arr):", np.modf(arr))                       # Splits into fractional and integral parts
"""(array([0.234, 0.5  , 0.5  ]), array([1., 2., 3.]))"""
"""__________________________________________________________________________________________________________________________________________"""

bool_arr1 = np.array([True, True, False, False])
bool_arr2 = np.array([True, False, True, False])

print("np.logical_and(bool_arr1, bool_arr2):", np.logical_and(bool_arr1, bool_arr2))      # Logical AND
"""[ True False False False]"""

print("np.logical_or(bool_arr1, bool_arr2):", np.logical_or(bool_arr1, bool_arr2))        # Logical OR
"""[ True  True  True False]"""

print("np.logical_not(bool_arr1):", np.logical_not(bool_arr1))                            # Logical NOT
"""[False False  True  True]"""

print("np.logical_xor(bool_arr1, bool_arr2):", np.logical_xor(bool_arr1, bool_arr2))      # Logical XOR
"""[False  True  True False]"""

print("np.where(bool_arr1, 1, 0):", np.where(bool_arr1, 1, 0))                          # Conditional selection
"""[1 1 0 0]""" 
"""__________________________________________________________________________________________________________________________________________"""
A = np.array([[1, 2], [3, 4]]) 
B = np.array([[5, 6], [7, 8]]) 

print("np.matmul(A, B):", np.matmul(A, B))   # Matrix multiplication
"""[[19 22] 
  [43 50]]""" 

print("np.inner(A, B):", np.inner(A, B))     # Inner product
"""[[19 22]   
  [43 50]]"""                                # Calculated as [[1*5 + 2*7, 1*6 + 2*8], [3*5 + 4*7, 3*6 + 4*8]]
  
print("np.outer(A, B):", np.outer(A, B))     # Outer product
"""[[ 5  6  7  8]   
  [10 12 14 16]   
  [15 18 21 24]   
  [20 24 28 32]]"""                          # Calculated as [[1*5, 1*6, 1*7, 1*8], [2*5, 2*6, 2*7, 2*8], [3*5, 3*6, 3*7, 3*8], [4*5, 4*6, 4*7, 4*8]]
  
print("np.dot(A, b):", np.dot(A, b))         # Dot product (matrix-vector multiplication)
"""[17 39]"""                                # Calculated as [1*4 + 2*5 + 3*6, 3*4 + 4*5 + 6*6]
  
print("np.trace(A):", np.trace(A))           # Trace of a matrix
"""5"""                                      # Calculated as 1 + 4
  
print("np.det(A):", np.linalg.det(A))        # Determinant of a matrix
"""-2.0"""                                   # Calculated as (1*4 - 2*3)
  
print("np.inv(A):", np.linalg.inv(A))        # Inverse of a matrix
"""[[-2.   1. ]   
  [ 1.5 -0.5]]"""                            # Calculated as (1/det(A)) * [[d, -b], [-c, a]] where A=[[a, b], [c, d]]
    
# Solve Ax = b 
A = np.array([[2, 1], [1, 3]]) 
b = np.array([5, 10]) 
x = np.linalg.solve(A, b) 
print("Solution x:", x)
"""[1. 3.]"""
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
"""__________________________________________________________________________________________________________________________________________"""

print("np.concatenate([a, b]):", np.concatenate([a, b]))
"""[1 2 3 2 2 2]"""

print("np.stack([a, b]):", np.stack([a, b]))
"""[ [1 2 3]
  [2 2 2] ]"""

print("np.hstack([a, b]):", np.hstack([a, b]))
"""[1 2 3 2 2 2]"""

print("np.vstack([a, b]):", np.vstack([a, b]))
"""[ [1 2 3]
  [2 2 2] ]"""

print("np.split(a, 3):", np.split(a, 3))
"""[array([1]), array([2]), array([3])]"""

print("np.hsplit(np.array([[1,2,3],[4,5,6]]), 3):", np.hsplit(np.array([[1,2,3],[4,5,6]]), 3))
"""[array([[1],
       [4]]), array([[2],
       [5]]), array([[3],
       [6]])]"""

print("np.vsplit(np.array([[1,2,3],[4,5,6]]), 2):", np.vsplit(np.array([[1,2,3],[4,5,6]]), 2))
"""[array([[1, 2, 3]]), array([[4, 5, 6]])]"""

"""__________________________________________________________________________________________________________________________________________"""

print("np.sign([-5, 0, 5]):", np.sign([-5, 0, 5]))
"""[-1  0  1]"""                                                # Sign function

print("np.clip(a, 1.5, 2.5):", np.clip(a, 1.5, 2.5))            # Clip values to a specified range
"""[1.5 2.  2.5]"""                                             # Values below 1.5 set to 1.5, above 2.5 set to 2.5

print("np.diff(a):", np.diff(a))                                # Discrete difference
"""[1 1]"""                                                     # Computes the difference between consecutive elements

print("np.gradient(a):", np.gradient(a))                        # Numerical gradient
"""[1. 1. 1.]"""                                                # Calculates the gradient (rate of change) of the array.

print("np.trapz(a):", np.trapz(a))                              # Trapezoidal integration
"""12.0"""                                                      # Calculated as (1+2)/2*1 + (2+3)/2*1 + (3+4)/2*1 + (4+5)/2*1 = 12.0
"""__________________________________________________________________________________________________________________________________________"""

# Random array

# Seed for reproducibility
np.random.seed(42)

# Random float in [0, 1)
print("np.random.rand():", np.random.rand())
"""0.3745401188473625"""

# Array of random floats
print("np.random.rand(3):", np.random.rand(3))
"""[0.95071431 0.73199394 0.59865848]"""

# 2D array of random floats
print("np.random.rand(2, 3):", np.random.rand(2, 3))
"""[[0.15601864 0.15599452 0.05808361]
 [0.86617615 0.60111501 0.70807258]]"""

 # Single random integer
print("np.random.randint(10):", np.random.randint(10))
"""4"""

# Array of random integers
print("np.random.randint(0, 10, size=5):", np.random.randint(0, 10, size=5))
"""[3 7 9 3 5]"""

# 2D array of random integers
print("np.random.randint(0, 10, size=(2, 3)):", np.random.randint(0, 10, size=(2, 3)))
"""[[3 5 2]
 [4 7 6]]"""

 # Random choice from array
arr = np.array([1, 2, 3, 4, 5])
print("np.random.choice(arr, size=3):", np.random.choice(arr, size=3))
"""[2 5 1]"""

# Random choice with probabilities
print("np.random.choice(arr, size=3, p=[0.1, 0.1, 0.1, 0.3, 0.4]):", 
      np.random.choice(arr, size=3, p=[0.1, 0.1, 0.1, 0.3, 0.4]))
"""[5 4 5]"""

# Random choice with replacement (default)
print("np.random.choice(arr, size=10, replace=True):", np.random.choice(arr, size=10, replace=True))
"""[1 5 3 4 5 5 1 5 3 4]"""

# Random choice without replacement
print("np.random.choice(arr, size=3, replace=False):", np.random.choice(arr, size=3, replace=False))
"""[1 4 2]"""

# Shuffle array (in-place)
shuffle_arr = np.arange(10)
np.random.shuffle(shuffle_arr)
print("Shuffled array:", shuffle_arr)
"""[8 1 5 0 7 2 9 4 3 6]"""

# Permutation (returns shuffled array)
print("np.random.permutation(5):", np.random.permutation(5))
"""[3 0 1 4 2]"""

print("np.random.permutation(arr):", np.random.permutation(arr))
"""[4 5 1 3 2]"""

# Random samples - uniform distribution over [0.0, 1.0)
print("np.random.random_sample():", np.random.random_sample())
"""0.44538719"""

print("np.random.random_sample((2, 3)):", np.random.random_sample((2, 3)))
"""[[0.093324 0.575946 0.929324]
 [0.318569 0.66741  0.131798]]"""
"""__________________________________________________________________________________________________________________________________________"""

arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([3, 4, 5, 6, 7])

print("np.intersect1d(arr1, arr2):", np.intersect1d(arr1, arr2))     # Finds common elements between two arrays
"""[3 4 5]"""

print("np.union1d(arr1, arr2):", np.union1d(arr1, arr2))             # Combines unique elements from both arrays
"""[1 2 3 4 5 6 7]"""

print("np.setdiff1d(arr1, arr2):", np.setdiff1d(arr1, arr2))         # Elements in arr1 not in arr2
"""[1 2]"""

print("np.setxor1d(arr1, arr2):", np.setxor1d(arr1, arr2))           # Elements in either arr1 or arr2 but not in both
"""[1 2 6 7]"""

print("np.in1d(arr1, arr2):", np.in1d(arr1, arr2))                   # Checks which elements of arr1 are in arr2
"""[False False  True  True  True]"""

print("np.isin(arr1, arr2):", np.isin(arr1, arr2))                   # Similar to in1d but returns an array of the same shape as arr1
"""[False False  True  True  True]"""

"""__________________________________________________________________________________________________________________________________________"""

# np.char.add() - Element-wise string concatenation
print("np.char.add(['Hello, ', 'Good '], ['World!', 'Morning!']):", 
      np.char.add(['Hello, ', 'Good '], ['World!', 'Morning!']))
"""['Hello, World!' 'Good Morning!']"""

# np.char.multiply() - Repeat strings
print("np.char.multiply('Hi! ', 3):", np.char.multiply('Hi! ', 3))
"""'Hi! Hi! Hi! '"""

# np.char.capitalize() - Capitalize first letter
print("np.char.capitalize(['hello world', 'good morning']):", 
      np.char.capitalize(['hello world', 'good morning']))
"""['Hello world' 'Good morning']"""

# np.char.title() - Title case
print("np.char.title(['hello world', 'good morning']):", 
      np.char.title(['hello world', 'good morning']))
"""['Hello World' 'Good Morning']"""

# np.char.lower() - Convert to lowercase
print("np.char.lower(['HELLO', 'WORLD']):", 
      np.char.lower(['HELLO', 'WORLD']))
"""['hello' 'world']"""

# np.char.upper() - Convert to uppercase
print("np.char.upper(['hello', 'world']):", 
      np.char.upper(['hello', 'world']))
"""['HELLO' 'WORLD']"""

# np.char.split() - Split strings
print("np.char.split(['Hello World', 'Good Morning']):", 
      np.char.split(['Hello World', 'Good Morning']))
"""[list(['Hello', 'World']) list(['Good', 'Morning'])]"""

# np.char.join() - Join characters with a separator
print("np.char.join('-', ['2024', '06', '15']):", 
      np.char.join('-', ['2024', '06', '15']))
"""['2-0-2-4' '0-6' '1-5']"""

# np.char.strip() - Remove leading/trailing characters
print("np.char.strip(['  Hello  ', '  World  ']):", 
      np.char.strip(['  Hello  ', '  World  ']))
"""['Hello' 'World']"""

# np.char.replace() - Replace substring
print("np.char.replace(['Hello World', 'Good Morning'], 'o', '0'):", 
      np.char.replace(['Hello World', 'Good Morning'], 'o', '0'))
"""['Hell0 W0rld' 'G00d M0rning']"""

"""__________________________________________________________________________________________________________________________________________"""

A = np.array([[1, 2], [3, 4]])

# np.linalg.matrix_power() - Raise matrix to power
print("np.linalg.matrix_power(A, 2):", np.linalg.matrix_power(A, 2))
"""[[ 7 10]
  [15 22]]"""                                                            # Calculated as A * A

# np.linalg.matrix_rank() - Matrix rank
print("np.linalg.matrix_rank(A):", np.linalg.matrix_rank(A))
"""2"""                                                                  # Rank of matrix A

# np.convolve() - Convolution
print("np.convolve([1, 2, 3], [0, 1, 0.5], mode='full'):", 
      np.convolve([1, 2, 3], [0, 1, 0.5], mode='full'))
"""[0.  1.  2.5 4.  1.5]"""                                              # Full convolution of two sequences

# np.correlate() - Correlation
print("np.correlate([1, 2, 3], [0, 1, 0.5], mode='full'):", 
      np.correlate([1, 2, 3], [0, 1, 0.5], mode='full'))
"""[1.5 4.  2.5 1.  0. ]"""                                              # Full correlation of two sequences

# np.corrcoef() - Correlation coefficient
print("np.corrcoef([1, 2, 3], [1, 2, 3]):", 
      np.corrcoef([1, 2, 3], [1, 2, 3]))
"""[[1. 1.]
  [1. 1.]]"""                                                            # Correlation coefficient matrix

# np.cov() - Covariance matrix
print("np.cov([1, 2, 3], [1, 2, 3]):", 
      np.cov([1, 2, 3], [1, 2, 3]))
"""[[1. 1.]
  [1. 1.]]"""                                                            # Covariance matrix

# np.linalg.eig() - Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues of A:", eigenvalues)
"""[-0.37228132  5.37228132]"""
print("Eigenvectors of A:\n", eigenvectors)
"""[[-0.82456484 -0.41597356]
  [ 0.56576746 -0.90937671]]"""

# np.linalg.eigh() - Eigenvalues/eigenvectors of Hermitian matrix
eigenvalues_h, eigenvectors_h = np.linalg.eigh(np.array([[2, 1], [1, 2]]))
print("Eigenvalues of Hermitian matrix:", eigenvalues_h)
"""[1. 3.]"""
print("Eigenvectors of Hermitian matrix:\n", eigenvectors_h)
"""[[-0.70710678 -0.70710678]
  [ 0.70710678 -0.70710678]]"""

# np.linalg.svd() - Singular value decomposition
U, S, VT = np.linalg.svd(A)
print("U matrix from SVD:\n", U)
"""[[ -0.57604844 -0.81741556]
  [ -0.81741556  0.57604844]]"""
print("Singular values from SVD:", S)
"""[5.4649857  0.36596619]"""
print("VT matrix from SVD:\n", VT)
"""[[-0.57604844 -0.81741556]
  [ 0.81741556 -0.57604844]]"""

# np.linalg.qr() - QR decomposition
Q, R = np.linalg.qr(A)
print("Q matrix from QR decomposition:\n", Q)
"""[[-0.31622777 -0.9486833 ]
  [-0.9486833   0.31622777]]"""
print("R matrix from QR decomposition:\n", R)
"""[[-3.16227766 -4.42718872]
  [ 0.          0.63245553]]"""

# np.linalg.cholesky() - Cholesky decomposition
L = np.linalg.cholesky(np.array([[4, 2], [2, 3]]))
print("L matrix from Cholesky decomposition:\n", L)
"""[[2.         0.        ]
  [1.         1.41421356]]"""

# np.linalg.norm() - Matrix or vector norm
print("np.linalg.norm(a):", np.linalg.norm(a))
"""3.7416573867739413"""                                                 # Euclidean norm (L2 norm)

# np.linalg.cond() - Condition number
print("np.linalg.cond(A):", np.linalg.cond(A))
"""14.933034373659268"""                                                 # Condition number of matrix A

# np.linalg.tensorsolve() - Solve tensor equation
B = np.array([[5, 6], [7, 8]])
X = np.linalg.tensorsolve(A, B)
print("Solution X for tensor equation AX = B:\n", X)
"""[[ -4.   5.]
  [  6.5 -7.]]"""

"""__________________________________________________________________________________________________________________________________________"""

# np.pi
print("Value of np.pi:", np.pi)
"""3.141592653589793"""

# np.e
print("Value of np.e:", np.e)
"""2.718281828459045"""

# np.inf
print("Value of np.inf:", np.inf)
"""inf"""

# np.nan
print("Value of np.nan:", np.nan)
"""nan"""                                                  # Not a Number

# np.NINF (negative infinity)
print("Value of np.NINF:", np.NINF)
"""-inf"""

# np.NZERO (negative zero)
print("Value of np.NZERO:", np.NZERO)
"""-0.0"""

# np.PZERO (positive zero)
print("Value of np.PZERO:", np.PZERO)
"""0.0"""

# np.euler_gamma
print("Value of np.euler_gamma:", np.euler_gamma)          # Euler-Mascheroni constant
"""0.5772156649015329"""

# np.newaxis
arr = np.array([1, 2, 3])
print("Using np.newaxis to increase dimensions:", arr[:, np.newaxis])
"""[[1]
  [2]
  [3]]"""

# np.nanmean() - Mean ignoring NaNs
arr_with_nan = np.array([1, 2, np.nan, 4])
print("np.nanmean(arr_with_nan):", np.nanmean(arr_with_nan))
"""2.3333333333333335"""                                     # Mean of [1, 2, 4] ignoring NaN

# np.nanstd() - Standard deviation ignoring NaNs
print("np.nanstd(arr_with_nan):", np.nanstd(arr_with_nan))
"""1.247219128924647"""                                      # Std dev of [1, 2, 4] ignoring NaN

# np.nanvar() - Variance ignoring NaNs
print("np.nanvar(arr_with_nan):", np.nanvar(arr_with_nan))
"""1.5555555555555556"""                                     # Variance of [1, 2, 4] ignoring NaN

"""__________________________________________________________________________________________________________________________________________"""
