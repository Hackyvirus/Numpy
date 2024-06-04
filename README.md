# NumPy Detailed Guide

## 1. Introduction to NumPy

### Installation and Importing

NumPy is a fundamental package for scientific computing in Python. It is widely used for its powerful n-dimensional array object and functions for manipulating these arrays. To get started with NumPy, you need to install it. You can do this using pip, which is a package manager for Python:

```sh
pip install numpy
```

Once installed, you can import NumPy into your Python environment. It is common practice to import it using the alias `np`:

```python
import numpy as np
```

This shorthand makes your code more readable and easier to type.

## 2. Creating Arrays

### From Lists and Tuples

NumPy arrays can be created from Python lists or tuples using the `np.array` function. This allows you to leverage NumPy's optimized array operations on existing data structures:

```python
arr = np.array([1, 2, 3, 4, 5])
print(arr)  # Output: [1 2 3 4 5]

arr_2d = np.array(((1, 2, 3), (4, 5, 6)))
print(arr_2d)
# Output:
# [[1 2 3]
#  [4 5 6]]
```

### Using Functions

NumPy provides a variety of functions to create arrays of specific shapes and values:

- **`arange`**: Generates arrays with regularly spaced values.

```python
arr = np.arange(0, 10, 2)
print(arr)  # Output: [0 2 4 6 8]
```

- **`linspace`**: Creates arrays with a specified number of evenly spaced values between two points.

```python
arr = np.linspace(0, 1, 5)
print(arr)  # Output: [0.   0.25 0.5  0.75 1.  ]
```

- **`zeros`**, **`ones`**, **`empty`**, and **`full`**: Create arrays filled with 0s, 1s, uninitialized values, or a specified value.

```python
zeros = np.zeros((2, 3))
print(zeros)
# Output:
# [[0. 0. 0.]
#  [0. 0. 0.]]

ones = np.ones((2, 3))
print(ones)
# Output:
# [[1. 1. 1.]
#  [1. 1. 1.]]

empty = np.empty((2, 3))
print(empty)  # Output: Uninitialized values, random

full = np.full((2, 3), 7)
print(full)
# Output:
# [[7 7 7]
#  [7 7 7]]
```

- **Identity and Diagonal Matrices**: Create identity matrices and diagonal matrices.

```python
identity = np.eye(3)
print(identity)
# Output:
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

diag = np.diag([1, 2, 3])
print(diag)
# Output:
# [[1 0 0]
#  [0 2 0]
#  [0 0 3]]
```

### Random Values

NumPy also provides a submodule `np.random` for generating arrays with random values:

- **Uniform distribution**: Random values between 0 and 1.

```python
random_values = np.random.random((2, 3))
print(random_values)
# Output: Random values between 0 and 1
```

- **Normal distribution**: Random values from a normal (Gaussian) distribution.

```python
randn_values = np.random.randn(2, 3)
print(randn_values)
# Output: Random values from a normal distribution
```

- **Random integers**: Random integers within a specified range.

```python
randint_values = np.random.randint(0, 10, (2, 3))
print(randint_values)
# Output: Random integers between 0 and 9
```

## 3. Array Attributes

NumPy arrays come with several built-in attributes that provide useful information about the array:

- **`shape`**: Tuple representing the dimensions of the array.

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.shape)  # Output: (2, 3)
```

- **`ndim`**: Number of dimensions of the array.

```python
print(arr.ndim)  # Output: 2
```

- **`size`**: Total number of elements in the array.

```python
print(arr.size)  # Output: 6
```

- **`dtype`**: Data type of the array elements.

```python
print(arr.dtype)  # Output: int64 (or int32 depending on the system)
```

- **`itemsize`**: Size in bytes of each element.

```python
print(arr.itemsize)  # Output: 8 (for int64)
```

## 4. Array Indexing and Slicing

### Basic Indexing

Accessing individual elements of an array is similar to Python lists:

```python
arr = np.array([1, 2, 3, 4, 5])
print(arr[0])   # Output: 1
print(arr[-1])  # Output: 5
```

For multidimensional arrays, you can use a tuple of indices:

```python
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print(arr_2d[0, 1])  # Output: 2
```

### Slicing

Slicing allows you to access subarrays. The syntax is similar to Python lists:

```python
arr = np.array([1, 2, 3, 4, 5])
print(arr[1:4])  # Output: [2 3 4]
print(arr[:3])   # Output: [1 2 3]
print(arr[::2])  # Output: [1 3 5]
```

For multidimensional arrays, you can slice along each axis:

```python
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print(arr_2d[:, 1])  # Output: [2 5]
print(arr_2d[1, :])  # Output: [4 5 6]
```

### Boolean Indexing

You can use boolean conditions to filter arrays:

```python
arr = np.array([1, 2, 3, 4, 5])
print(arr[arr > 2])  # Output: [3 4 5]
```

### Fancy Indexing

Fancy indexing allows you to access multiple array elements at once using an array of indices:

```python
arr = np.array([1, 2, 3, 4, 5])
indices = [0, 2, 4]
print(arr[indices])  # Output: [1 3 5]
```

## 5. Array Manipulation

### Reshaping Arrays

You can change the shape of an array without modifying its data using the `reshape` method:

```python
arr = np.array([1, 2, 3, 4, 5, 6])
reshaped = arr.reshape((2, 3))
print(reshaped)
# Output:
# [[1 2 3]
#  [4 5 6]]
```

### Flattening Arrays

Flattening an array converts it to a 1D array:

```python
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
flattened = arr_2d.flatten()
print(flattened)  # Output: [1 2 3 4 5 6]
```

### Transposing Arrays

Transposing an array swaps its axes:

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
transposed = arr.T
print(transposed)
# Output:
# [[1 4]
#  [2 5]
#  [3 6]]
```

### Concatenating and Stacking Arrays

You can concatenate arrays along a specified axis using `np.concatenate`:

```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
concatenated = np.concatenate((arr1, arr2))
print(concatenated)  # Output: [1 2 3 4 5 6]
```

To stack arrays along a new axis, use `np.stack`:

```python
stacked = np.stack((arr1, arr2), axis=0)
print(stacked)
# Output:
# [[1 2 3]
#  [4 5 6]]
```

### Splitting Arrays

You can split an array into multiple subarrays using `np.split`:

```python
arr = np.array([1, 2, 3, 4, 5, 6])
split = np.split(arr, 3)
print(split)  # Output: [array([1, 2]), array([3

, 4]), array([5, 6])]
```

## 6. Universal Functions (ufuncs)

NumPy provides a set of functions that operate element-wise on arrays, known as universal functions or ufuncs.

### Arithmetic Operations

You can perform element-wise arithmetic operations using ufuncs:

```python
arr = np.array([1, 2, 3])
print(arr + 1)  # Output: [2 3 4]
print(arr * 2)  # Output: [2 4 6]
print(arr ** 2) # Output: [1 4 9]
```

### Trigonometric Functions

NumPy provides trigonometric functions like `sin`, `cos`, and `tan`:

```python
arr = np.array([0, np.pi/2, np.pi])
print(np.sin(arr))  # Output: [0. 1. 0.]
```

### Exponential and Logarithmic Functions

You can use functions like `exp`, `log`, and `log10`:

```python
arr = np.array([1, 2, 3])
print(np.exp(arr))   # Output: [ 2.71828183  7.3890561  20.08553692]
print(np.log(arr))   # Output: [0.         0.69314718 1.09861229]
print(np.log10(arr)) # Output: [0.         0.30103    0.47712125]
```

## 7. Aggregate Functions

NumPy includes various functions for computing aggregate values across arrays.

### Sum, Mean, Median, Standard Deviation, and Variance

You can compute common statistical measures using NumPy:

```python
arr = np.array([1, 2, 3, 4, 5])
print(np.sum(arr))   # Output: 15
print(np.mean(arr))  # Output: 3.0
print(np.median(arr))# Output: 3.0
print(np.std(arr))   # Output: 1.4142135623730951
print(np.var(arr))   # Output: 2.0
```

### Min, Max, Argmin, Argmax

Find the minimum, maximum, and their indices:

```python
print(np.min(arr))    # Output: 1
print(np.max(arr))    # Output: 5
print(np.argmin(arr)) # Output: 0
print(np.argmax(arr)) # Output: 4
```

### Cumulative Sum and Product

Compute the cumulative sum and product of array elements:

```python
print(np.cumsum(arr))  # Output: [ 1  3  6 10 15]
print(np.cumprod(arr)) # Output: [ 1  2  6 24 120]
```

## 8. Linear Algebra

NumPy provides a submodule `np.linalg` for linear algebra operations.

### Dot Product and Matrix Multiplication

Compute dot products and matrix multiplications:

```python
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])
print(np.dot(arr1, arr2))
# Output:
# [[19 22]
#  [43 50]]
```

### Solving Linear Systems

Solve linear equations using `np.linalg.solve`:

```python
A = np.array([[1, 2], [3, 4]])
b = np.array([5, 6])
x = np.linalg.solve(A, b)
print(x)  # Output: [-4.   4.5]
```

### Eigenvalues and Eigenvectors

Compute eigenvalues and eigenvectors:

```python
eigvals, eigvecs = np.linalg.eig(A)
print(eigvals)  # Output: [-0.37228132  5.37228132]
print(eigvecs)
# Output:
# [[-0.82456484 -0.41597356]
#  [ 0.56576746 -0.90937671]]
```

### Determinant and Inverse of a Matrix

Compute the determinant and inverse of a matrix:

```python
print(np.linalg.det(A))    # Output: -2.0000000000000004
print(np.linalg.inv(A))
# Output:
# [[-2.   1. ]
#  [ 1.5 -0.5]]
```

## 9. Random Number Generation

NumPy provides robust random number generation capabilities in the `np.random` module.

### Generating Random Numbers

Generate random numbers from various distributions:

```python
uniform_random = np.random.rand(2, 3)
print(uniform_random)  # Output: Random values between 0 and 1

normal_random = np.random.randn(2, 3)
print(normal_random)  # Output: Random values from a normal distribution
```

### Random Sampling

Random sampling from arrays:

```python
arr = np.array([1, 2, 3, 4, 5])
sample = np.random.choice(arr, 3)
print(sample)  # Output: Random sample of 3 elements from arr
```

### Setting Random Seed for Reproducibility

Set the random seed to ensure reproducibility:

```python
np.random.seed(42)
print(np.random.rand())  # Output: Same value every time this code is run
```

## 10. Broadcasting

Broadcasting allows NumPy to perform element-wise operations on arrays of different shapes.

### Understanding Broadcasting Rules

Broadcasting follows specific rules to match array shapes:

1. If the arrays have different numbers of dimensions, the shape of the smaller-dimensional array is padded with ones on its left side.
2. If the shape of the arrays in a dimension is different, the array with shape 1 in that dimension is stretched to match the other shape.
3. If the shapes are incompatible, an error is raised.

### Applying Broadcasting in Arithmetic Operations

Example of broadcasting:

```python
arr = np.array([1, 2, 3])
print(arr + 1)  # Output: [2 3 4]

arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print(arr2d + arr)  # Output: [[2 4 6]
#                           [5 7 9]]
```

## 11. Advanced Indexing

Advanced indexing allows for more complex data manipulation.

### Boolean Masking

Use boolean arrays to select elements:

```python
arr = np.array([1, 2, 3, 4, 5])
mask = arr > 2
print(arr[mask])  # Output: [3 4 5]
```

### Indexing with Arrays of Indices

Select multiple elements using an array of indices:

```python
indices = [0, 2, 4]
print(arr[indices])  # Output: [1 3 5]
```

### `np.where` and `np.extract`

Conditional selection and extraction:

```python
arr = np.array([1, 2, 3, 4, 5])
print(np.where(arr > 2))  # Output: (array([2, 3, 4]),)
print(np.extract(arr > 2, arr))  # Output: [3 4 5]
```

## 12. File I/O

NumPy provides functions for saving and loading arrays from files.

### Saving and Loading Arrays

Save and load arrays in binary format using `np.save` and `np.load`:

```python
arr = np.array([1, 2, 3, 4, 5])
np.save('array.npy', arr)
loaded_arr = np.load('array.npy')
print(loaded_arr)  # Output: [1 2 3 4 5]
```

### Reading and Writing Text Files

Save and load arrays in text format using `np.savetxt` and `np.loadtxt`:

```python
np.savetxt('array.txt', arr)
loaded_txt_arr = np.loadtxt('array.txt')
print(loaded_txt_arr)  # Output: [1. 2. 3. 4. 5.]
```

### Reading and Writing Binary Files

Use `np.fromfile` and `np.tofile` for raw binary I/O:

```python
arr.tofile('array.bin')
loaded_bin_arr = np.fromfile('array.bin', dtype=np.int64)
print(loaded_bin_arr)  # Output: [1 2 3 4 5]
```

## 13. Performance Optimization

NumPy is optimized for performance, but there are techniques to make it even faster.

### Vectorization

Vectorize operations to avoid Python loops:

```python
arr = np.array([1, 2, 3, 4, 5])
squared = arr ** 2
print(squared)  # Output: [ 1  4  9 16 25]
```

### Using `numba` for Just-in-Time Compilation

`numba` can JIT-compile Python functions for faster execution:

```python
from numba import jit

@jit
def sum_elements(arr):
    total = 0
    for i in arr:
        total += i
    return total

arr = np.array([1, 2, 3, 4, 5])
print(sum_elements(arr))  # Output: 15
```

### Memory Layout and Strides

Understanding and optimizing memory layout can improve performance:

```python
arr = np.array([[1, 2, 3

], [4, 5, 6]])
print(arr.strides)  # Output: (24, 8) on a 64-bit system
```

## 14. Structured Arrays

Structured arrays allow you to handle complex data types.

### Creating and Manipulating Structured Arrays

Define structured arrays with different data types for each field:

```python
dtype = [('name', 'S10'), ('age', 'i4'), ('height', 'f4')]
data = [('Alice', 25, 5.5), ('Bob', 30, 6.0)]
structured_arr = np.array(data, dtype=dtype)
print(structured_arr)
# Output: [(b'Alice', 25, 5.5) (b'Bob', 30, 6. )]
```

### Accessing Fields of Structured Arrays

Access specific fields using their names:

```python
print(structured_arr['name'])  # Output: [b'Alice' b'Bob']
print(structured_arr['age'])   # Output: [25 30]
```

## 15. Special Functions

NumPy includes several special functions for advanced use cases.

### Polynomial Functions

Handle polynomials using `np.poly1d`:

```python
p = np.poly1d([1, -2, 1])
print(p(0))  # Output: 1
print(p.roots)  # Output: [1. 1.]
```

### Sorting, Searching, and Counting Functions

Sort, search, and count elements:

```python
arr = np.array([3, 1, 2, 5, 4])
print(np.sort(arr))  # Output: [1 2 3 4 5]
print(np.argsort(arr))  # Output: [1 2 0 4 3]
print(np.count_nonzero(arr > 2))  # Output: 3
```

### Set Operations

Perform set operations like union, intersection, and difference:

```python
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([4, 5, 6, 7, 8])
print(np.union1d(arr1, arr2))       # Output: [1 2 3 4 5 6 7 8]
print(np.intersect1d(arr1, arr2))   # Output: [4 5]
print(np.setdiff1d(arr1, arr2))     # Output: [1 2 3]
```
