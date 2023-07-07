"""

    py -m venv matplotlib
    source matplotlib/Scripts/activate
    pip install matplotlib
    pip install numpy ##### ya instalado con matplotlib

"""
import numpy as np

'''
###### array como una lista ######
arr1 = np.array([1, 2, 3, 4, 5])

print(arr1) #[1 2 3 4 5]

print(np.__version__) #1.25.0

print(type(arr1)) #<class 'numpy.ndarray'>

###### array como una tupla ######
arr2 = np.array((1, 2, 3, 4, 5))

print(arr2) #se imprime exactamente lo mismo que en el caso de una lista: [1 2 3 4 5]

###### 0-D Arrays: números escalares ######
arr3 = np.array(42)

print(arr3) #42

###### 1-D Arrays: vectores ######
arr4 = np.array([1, 2, 3, 4, 5])

print(arr4) #[1 2 3 4 5]

###### 2-D Arrays: matrices ######
arr5 = np.array([[1, 2, 3], [4, 5, 6]])

print(arr5) 
#[[1 2 3]
# [4 5 6]]

###### 3-D Arrays: matrices 3D ######
arr6 = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

print(arr6)
# [[[1 2 3]
#   [4 5 6]]
#
#  [[1 2 3]
#   [4 5 6]]]

###### Evaluar las dimensiones de una matriz ######
a = np.array(42)
b = np.array([1, 2, 3, 4, 5])
c = np.array([[1, 2, 3], [4, 5, 6]])
d = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

print(a.ndim) #0
print(b.ndim) #1
print(c.ndim) #2
print(d.ndim) #3

###### Fijar las dimensiones de una matriz ######
arr7 = np.array([1, 2, 3, 4], ndmin=5)

print(arr7) #[[[[[1 2 3 4]]]]]
print('number of dimensions :', arr7.ndim) #number of dimensions : 5


###### Acceso a cualquier elemento de un arrar de NumPy ######
#Se realiza como en cualquier otra lista de python: acordarse que el primer lugar comienza por el cero
arr8 = np.array([1, 2, 3, 4])

print(arr8[0]) #1
print(arr8[1]) #2

#sumar elementos de un array de NumPy:
print(arr8[2] + arr8[3]) #7

###### Acceso a un elemento de un array 2D (una matriz) de NumPy ######
arr9 = np.array([[1,2,3,4,5], [6,7,8,9,10]])

print(arr9)
#[[ 1  2  3  4  5]
# [ 6  7  8  9 10]]
print('2do elemento en 1er fila: ', arr9[0, 1]) #2do elemento en 1er fila:  2
print('3er elemento en 2da fila: ', arr9[1, 2]) #3er elemento en 2da fila:  8

###### Acceso a un elemento de un array 3D (una matriz 3D) de NumPy ######
arr10 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

print(arr10)
# [[[ 1  2  3]
#   [ 4  5  6]]
#
#  [[ 7  8  9]
#   [10 11 12]]]
print(arr10[0, 1, 2]) #6

###### Acceso a un elemento mediante un número negativo ######
arr11 = np.array([[1,2,3,4,5], [6,7,8,9,10]])

print(arr11)
# [[ 1  2  3  4  5]
#  [ 6  7  8  9 10]]
print('Last element from 2nd dim: ', arr11[1, -1]) # Last element from 2nd dim:  10

###### Slice: seccionamiento de una matriz ######
arr12 = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr12) #[1 2 3 4 5 6 7]
print(arr12[1:5]) #[2 3 4 5]
#Note: The result includes the start index, but excludes the end index.

print(arr12[4:]) #[5 6 7]
print(arr12[:4]) #[1 2 3 4]
print(arr12[-3:-1]) #[5 6]

###### Step: se puede especificar un paso mediante el cual se secciona la matriz ######

arr13 = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr13[1:5:2]) #[2 4]
print(arr13[::2]) #[1 3 5 7]

###### Seccionamiento de una matriz 2D ######
arr14 = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

print(arr14)
# [[ 1  2  3  4  5]
#  [ 6  7  8  9 10]]
print(arr14[1, 1:4]) # [7 8 9]

###### Retornar el tercer elemento de ambas submatrices ######
arr15 = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

print(arr15)
# [[ 1  2  3  4  5]
#  [ 6  7  8  9 10]]
print(arr15[0:2, 2]) # [3 8]

###### Retornar desde el segundo al cuarto elemento de ambas submatrices (se forma una matriz 2D también) ######
print(arr15[0:2, 1:4])
# [[2 3 4]
#  [7 8 9]]
'''

########################################################################
########################################################################


"""
    Data Types in NumPy
NumPy has some extra data types, and refer to data types with one character, like i for integers, u for unsigned integers etc.

Below is a list of all data types in NumPy and the characters used to represent them.

In NumPy 1.25.0, the following data types are available:

# bool: Boolean (True or False) stored as a byte
# int8: 8-bit integer (-128 to 127)
# int16: 16-bit integer (-32768 to 32767)
# int32: 32-bit integer (-2147483648 to 2147483647)
# int64: 64-bit integer (-9223372036854775808 to 9223372036854775807)
# uint8: 8-bit unsigned integer (0 to 255)
# uint16: 16-bit unsigned integer (0 to 65535)
# uint32: 32-bit unsigned integer (0 to 4294967295)
# uint64: 64-bit unsigned integer (0 to 18446744073709551615)
# float16: Half-precision floating point
# float32: Single-precision floating point
# float64: Double-precision floating point
# complex64: Complex number represented by two 32-bit floats
# complex128: Complex number represented by two 64-bit floats

"""

#######################################################
#######################################################
"""
###### Get the data type of an array object: ######

arr1 = np.array([1, 2, 3, 4])

print(arr1.dtype) #int32

arr2 = np.array(['apple', 'banana', 'cherry'])

print(arr2.dtype) #<U6

a = np.array([2, 3, 4])
print(a.dtype) #int32

###### The Difference Between Copy and View: ######
arr_copy = np.array([1, 2, 3, 4, 5])
x_copy = arr_copy.copy()
arr_copy[0] = 42

print(arr_copy) #[42  2  3  4  5]
print(x_copy) #[1 2 3 4 5]

#The copy SHOULD NOT be affected by the changes made to the original array.

arr_view = np.array([1, 2, 3, 4, 5])
x_view = arr_view.view()
arr_view[0] = 42

print(arr_view) #[42  2  3  4  5]
print(x_view) #[42  2  3  4  5]

#The view SHOULD be affected by the changes made to the original array.

##########################################################

#Make a view, change the view, and display both arrays:
arr_view2 = np.array([1, 2, 3, 4, 5])
x_view2 = arr_view2.view()
x_view2[0] = 31

print(arr_view2) #[31  2  3  4  5]
print(x_view2) #[31  2  3  4  5]

#The original array SHOULD be affected by the changes made to the view.

##########################################################

# Copies owns the data, and views does not own the data, but how can we check this?
# Every NumPy array has the attribute base that returns None if the array owns the data.

arr_base = np.array([1, 2, 3, 4, 5])

x_base = arr_base.copy()
y_base = arr_base.view()

print(x_base.base) #None (the array owns the data)
print(y_base.base) #[1 2 3 4 5] (the array doesn't owns the data)

# The copy returns None.
# The view returns the original array.


###### Get the shape of an array object: ######

#The shape of an array is the number of elements in each dimension.

arr_shape = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

print(arr_shape.shape) #(2, 4) matriz de dos filas y cuatro columnas

arr_shape2 = np.array([1, 2, 3, 4], ndmin=5)

print(arr_shape2)
print('Shape of array :', arr_shape2.shape) #Shape of array : (1, 1, 1, 1, 4)

###### Reshaping arrays: ######

# Reshaping means changing the shape of an array.
# The shape of an array is the number of elements in each dimension.
# By reshaping we can add or remove dimensions or change number of elements in each dimension.

arr_reshape = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

newarr = arr_reshape.reshape(4, 3)

print(newarr)
# [[ 1  2  3]
#  [ 4  5  6]
#  [ 7  8  9]
#  [10 11 12]]

newarr2 = arr_reshape.reshape(2, 3, 2)

print(newarr2)
# [[[ 1  2]
#   [ 3  4]
#   [ 5  6]]

#  [[ 7  8]
#   [ 9 10]
#   [11 12]]]

# Can We Reshape Into any Shape?
# Yes, as long as the elements required for reshaping are equal in both shapes.
# We can reshape an 8 elements 1D array into 4 elements in 2 rows 2D array but we cannot reshape it into a 3 elements 3 rows 2D array as that would require 3x3 = 9 elements.

#Check if the returned array is a copy or a view:
arr_reshape2 = np.array([1, 2, 3, 4, 5, 6, 7, 8])

print(arr_reshape2.reshape(2, 4).base) #[1 2 3 4 5 6 7 8] (it's a view)
#The example above returns the original array, so it is a view.


###### Iterating Arrays: ######

# Iterating means going through elements one by one.

# As we deal with multi-dimensional arrays in numpy, we can do this using basic for loop of python.

# If we iterate on a 1-D array it will go through each element one by one.

arr_for = np.array([1, 2, 3])

for x in arr_for:
    print(x)
# 1
# 2
# 3

#In a 2-D array it will go through all the rows.

arr_for2 = np.array([[1, 2, 3], [4, 5, 6]])

for x in arr_for2:  
    print(x)
# [1 2 3]
# [4 5 6]

#######################################################
# To return the actual values, the scalars, we have to iterate the arrays in each dimension.

for x in arr_for2:
    for y in x:
        print(y)
# 1
# 2
# 3
# 4
# 5
# 6

#######################################################
#Iterating 3-D Arrays

arr_for3 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

for x in arr_for3:
    print(x)
# [[1 2 3]
#  [4 5 6]]
# [[ 7  8  9]
#  [10 11 12]]

for x in arr_for3:
    for y in x:
        for z in y:
            print(z)
# 1
# 2
# 3
# 4
# 5
# 6
# 7
# 8
# 9
# 10
# 11
# 12

"""
#######################################################
#######################################################
"""
###### Iterating Arrays Using nditer(): ######

# The function nditer() is a helping function that can be used from very basic to very advanced iterations. It solves some basic issues which we face in iteration, lets go through it with examples.

#######################################################
# Iterating on Each Scalar Element
# In basic for loops, iterating through each scalar of an array we need to use n for loops which can be difficult to write for arrays with very high dimensionality.

# Example
# Iterate through the following 3-D array:

arr_nditer1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

for x in np.nditer(arr_nditer1):
    print(x)
# 1
# 2
# 3
# 4
# 5
# 6
# 7
# 8

#######################################################
# Iterating Array With Different Data Types
# We can use op_dtypes argument and pass it the expected datatype to change the datatype of elements while iterating.

# NumPy does not change the data type of the element in-place (where the element is in array) so it needs some other space to perform this action, that extra space is called buffer, and in order to enable it in nditer() we pass flags=['buffered'].

# Example
# Iterate through the array as a string:

arr_nditer2 = np.array([1, 2, 3])

for x in np.nditer(arr_nditer2, flags=['buffered'], op_dtypes=['S']):
    print(x)
# b'1'
# b'2'
# b'3'

#######################################################
# Iterating With Different Step Size
# We can use filtering and followed by iteration.

# Example
# Iterate through every scalar element of the 2D array skipping 1 element:


arr_nditer3 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

for x in np.nditer(arr_nditer3[:, ::2]):
    print(x)
# 1
# 3
# 5
# 7

#######################################################
# Enumerated Iteration Using ndenumerate()
# Enumeration means mentioning sequence number of somethings one by one.

# Sometimes we require corresponding index of the element while iterating, the ndenumerate() method can be used for those usecases.

# Example
# Enumerate on following 1D arrays elements:

arr_nditer4 = np.array([1, 2, 3])

for idx, x in np.ndenumerate(arr_nditer4):
    print(idx, x)
# (0,) 1
# (1,) 2
# (2,) 3

# Example
# Enumerate on following 2D array's elements:

arr_nditer5 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

for idx, x in np.ndenumerate(arr_nditer5):
    print(idx, x)
# (0, 0) 1
# (0, 1) 2
# (0, 2) 3
# (0, 3) 4
# (1, 0) 5
# (1, 1) 6
# (1, 2) 7
# (1, 3) 8

"""
#######################################################
#######################################################
"""
###### NumPy Joining Array: ######

# Joining NumPy Arrays
# Joining means putting contents of two or more arrays in a single array.

# In SQL we join tables based on a key, whereas in NumPy we join arrays by axes.

# We pass a sequence of arrays that we want to join to the concatenate() function, along with the axis. If axis is not explicitly passed, it is taken as 0.

# Example
# Join two arrays

arr_join_1 = np.array([1, 2, 3])

arr_join_2 = np.array([4, 5, 6])

arr_join_1_2 = np.concatenate((arr_join_1, arr_join_2))

print('arr_join_1_2: ',arr_join_1_2) # arr_join_1_2:  [1 2 3 4 5 6]

# Example
# Join two 2-D arrays along rows (axis=1):

arr_join_3 = np.array([[1, 2], [3, 4]])

arr_join_4 = np.array([[5, 6], [7, 8]])

arr_join_3_4 = np.concatenate((arr_join_3, arr_join_4), axis=1)

print('arr_join_3_4: ',arr_join_3_4)
# arr_join_3_4:  [[1 2 5 6]
#  [3 4 7 8]]

#######################################################
# Joining Arrays Using Stack Functions
# Stacking is same as concatenation, the only difference is that stacking is done along a new axis.

# We can concatenate two 1-D arrays along the second axis which would result in putting them one over the other, ie. stacking.

# We pass a sequence of arrays that we want to join to the stack() method along with the axis. If axis is not explicitly passed it is taken as 0.

# Example

arr_join_5 = np.array([1, 2, 3])

arr_join_6 = np.array([4, 5, 6])

arr_join_5_6 = np.stack((arr_join_5, arr_join_6), axis=1)

print('arr_join_5_6: ',arr_join_5_6)
# arr_join_5_6:  [[1 4]
#  [2 5]
#  [3 6]]

#######################################################
# Stacking Along Rows
# NumPy provides a helper function: hstack() to stack along rows.

# Example

arr_join_7 = np.array([1, 2, 3])

arr_join_8 = np.array([4, 5, 6])

arr_join_7_8 = np.hstack((arr_join_7, arr_join_8))

print('arr_join_7_8: ',arr_join_7_8) #arr_join_7_8:  [1 2 3 4 5 6]

#######################################################
# Stacking Along Columns
# NumPy provides a helper function: vstack()  to stack along columns.

# Example

arr_join_9 = np.array([1, 2, 3])

arr_join_10 = np.array([4, 5, 6])

arr_join_9_10 = np.vstack((arr_join_9, arr_join_10))

print('arr_join_9_10: ',arr_join_9_10)
# arr_join_9_10:  [[1 2 3]
#  [4 5 6]]

#######################################################
# Stacking Along Height (depth)
# NumPy provides a helper function: dstack() to stack along height, which is the same as depth.

# Example

arr_join_11 = np.array([1, 2, 3])

arr_join_12 = np.array([4, 5, 6])

arr_join_11_12 = np.dstack((arr_join_11, arr_join_12))

print('arr_join_11_12: ',arr_join_11_12)
# arr_join_11_12:  [[[1 4]
#   [2 5]
#   [3 6]]]
"""

#######################################################
#######################################################
"""
###### NumPy Splitting Array: ######

# Splitting NumPy Arrays
# Splitting is reverse operation of Joining.

# Joining merges multiple arrays into one and Splitting breaks one array into multiple.

# We use array_split() for splitting arrays, we pass it the array we want to split and the number of splits.

# ExampleGet your own Python Server
# Split the array in 3 parts:

arr_split1 = np.array([1, 2, 3, 4, 5, 6])

newarr_split1 = np.array_split(arr_split1, 3)

print('newarr_split1: ',newarr_split1)
#newarr_split1:  [array([1, 2]), array([3, 4]), array([5, 6])]

# Note: The return value is a list containing three arrays.

# If the array has less elements than required, it will adjust from the end accordingly.

# Example
# Split the array in 4 parts:

arr_split2 = np.array([1, 2, 3, 4, 5, 6])

newarr_split2 = np.array_split(arr_split2, 4)

print('newarr_split2: ',newarr_split2)
# newarr_split2:  [array([1, 2]), array([3, 4]), array([5]), array([6])]

# Note: We also have the method split() available but it will not adjust the elements when elements are less in source array for splitting like in example above, array_split() worked properly but split() would fail.

#######################################################
# Split Into Arrays
# The return value of the array_split() method is an array containing each of the split as an array.

# If you split an array into 3 arrays, you can access them from the result just like any array element:

# Example
# Access the splitted arrays:

arr_split3 = np.array([1, 2, 3, 4, 5, 6])

newarr_split3 = np.array_split(arr_split3, 3)

print('newarr_split3[0]: ',newarr_split3[0]) #newarr_split3[0]:  [1 2]
print('newarr_split3[1]: ',newarr_split3[1]) #newarr_split3[1]:  [3 4]
print('newarr_split3[2]: ',newarr_split3[2]) #newarr_split3[2]:  [5 6]

#######################################################
# Splitting 2-D Arrays
# Use the same syntax when splitting 2-D arrays.

# Use the array_split() method, pass in the array you want to split and the number of splits you want to do.

# Example
# Split the 2-D array into three 2-D arrays.

arr_split4 = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])

newarr_split4 = np.array_split(arr_split4, 3)

print('newarr_split4: ',newarr_split4)
# newarr_split4:  [array([[1, 2],
#        [3, 4]]), array([[5, 6],
#        [7, 8]]), array([[ 9, 10],
#        [11, 12]])]

# The example above returns three 2-D arrays.

# Let's look at another example, this time each element in the 2-D arrays contains 3 elements.

# Example
# Split the 2-D array into three 2-D arrays.

arr_split5 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])

newarr_split5 = np.array_split(arr_split5, 3)

print('newarr_split5: ',newarr_split5)
# newarr_split5:  [array([[1, 2, 3],
#        [4, 5, 6]]), array([[ 7,  8,  9],
#        [10, 11, 12]]), array([[13, 14, 15],
#        [16, 17, 18]])]

# The example above returns three 2-D arrays.

# In addition, you can specify which axis you want to do the split around.

# The example below also returns three 2-D arrays, but they are split along the row (axis=1).

# Example
# Split the 2-D array into three 2-D arrays along rows.

arr_split6 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])

newarr_split6 = np.array_split(arr_split6, 3, axis=1)

print('newarr_split6: ',newarr_split6)
# newarr_split6:  [array([[ 1],
#        [ 4],
#        [ 7],
#        [10],
#        [13],
#        [16]]), array([[ 2],
#        [ 5],
#        [ 8],
#        [11],
#        [14],
#        [17]]), array([[ 3],
#        [ 6],
#        [ 9],
#        [12],
#        [15],
#        [18]])]

# An alternate solution is using hsplit() opposite of hstack()

# Example
# Use the hsplit() method to split the 2-D array into three 2-D arrays along rows.


arr_split7 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])

newarr_split7 = np.hsplit(arr_split7, 3)

print('newarr_split7: ',newarr_split7)
# newarr_split7:  [array([[ 1],
#        [ 4],
#        [ 7],
#        [10],
#        [13],
#        [16]]), array([[ 2],
#        [ 5],
#        [ 8],
#        [11],
#        [14],
#        [17]]), array([[ 3],
#        [ 6],
#        [ 9],
#        [12],
#        [15],
#        [18]])]

# Note: Similar alternates to vstack() and dstack() are available as vsplit() and dsplit().
"""
#######################################################
#######################################################
"""
###### NumPy Searching Arrays: ######

# Searching Arrays
# You can search an array for a certain value, and return the indexes that get a match.

# To search an array, use the where() method.

# ExampleGet your own Python Server
# Find the indexes where the value is 4:

arr_search1 = np.array([1, 2, 3, 4, 5, 4, 4])

x_search1 = np.where(arr_search1 == 4)

print('x_search1:')
print(x_search1) #(array([3, 5, 6], dtype=int64),)

#######################################################
# The example above will return a tuple: (array([3, 5, 6],)

# Which means that the value 4 is present at index 3, 5, and 6.

# Example
# Find the indexes where the values are even:

arr_search2 = np.array([1, 2, 3, 4, 5, 6, 7, 8])

x_search2 = np.where(arr_search2%2 == 0)

print('x_search2:')
print(x_search2) #(array([1, 3, 5, 7], dtype=int64),)

# Example
# Find the indexes where the values are odd:

arr_search3 = np.array([1, 2, 3, 4, 5, 6, 7, 8])

x_search3 = np.where(arr_search3%2 == 1)

print('x_search3:')
print(x_search3) #(array([0, 2, 4, 6], dtype=int64),)

#######################################################
# Search Sorted
# There is a method called searchsorted() which performs a binary search in the array, and returns the index where the specified value would be inserted to maintain the search order.

# The searchsorted() method is assumed to be used on sorted arrays.

# Example
# Find the indexes where the value 7 should be inserted:

arr_search4 = np.array([6, 7, 8, 9])

x_search4 = np.searchsorted(arr_search4, 7)

print('x_search4:')
print(x_search4) #1
# Example explained: The number 7 should be inserted on index 1 to remain the sort order.

# The method starts the search from the left and returns the first index where the number 7 is no longer larger than the next value.

#######################################################
# Search From the Right Side
# By default the left most index is returned, but we can give side='right' to return the right most index instead.

# Example
# Find the indexes where the value 7 should be inserted, starting from the right:

arr_search5 = np.array([6, 7, 8, 9])

x_search5 = np.searchsorted(arr_search5, 7, side='right')

print('x_search5:')
print(x_search5) #2

# Example explained: The number 7 should be inserted on index 2 to remain the sort order.

# The method starts the search from the right and returns the first index where the number 7 is no longer less than the next value.

#######################################################
# Multiple Values
# To search for more than one value, use an array with the specified values.

# Example
# Find the indexes where the values 2, 4, and 6 should be inserted:

arr_search6 = np.array([1, 3, 5, 7])

x_search6 = np.searchsorted(arr_search6, [2, 4, 6])

print('x_search6:')
print(x_search6) #[1 2 3]
# The return value is an array: [1 2 3] containing the three indexes where 2, 4, 6 would be inserted in the original array to maintain the order.
"""

#######################################################
#######################################################
"""
###### NumPy Sorting Arrays: ######
# Sorting Arrays
# Sorting means putting elements in an ordered sequence.

# Ordered sequence is any sequence that has an order corresponding to elements, like numeric or alphabetical, ascending or descending.

# The NumPy ndarray object has a function called sort(), that will sort a specified array.

# ExampleGet your own Python Server
# Sort the array:

arr_sort1 = np.array([3, 2, 0, 1])

print(arr_sort1) #[3 2 0 1]
print(np.sort(arr_sort1)) #[0 1 2 3]

# Note: This method returns a copy of the array, leaving the original array unchanged.

# You can also sort arrays of strings, or any other data type:

# Example
# Sort the array alphabetically:

arr_sort2 = np.array(['banana', 'cherry', 'apple'])

print(arr_sort2) #['banana' 'cherry' 'apple']
print(np.sort(arr_sort2)) #['apple' 'banana' 'cherry']

# Example
# Sort a boolean array:

arr_sort3 = np.array([True, False, True])

print(arr_sort3) #[ True False  True]
print(np.sort(arr_sort3)) #[False  True  True]

#######################################################
# Sorting a 2-D Array
# If you use the sort() method on a 2-D array, both arrays will be sorted:

# Example
# Sort a 2-D array:

arr_sort4 = np.array([[3, 2, 4], [5, 0, 1]])

print(arr_sort4) 
#[[3 2 4]
# [5 0 1]]
print(np.sort(arr_sort4)) 
#[[2 3 4]
# [0 1 5]]

"""
#######################################################
#######################################################

###### NumPy Filter Array: ######

# Filtering Arrays
# Getting some elements out of an existing array and creating a new array out of them is called filtering.

# In NumPy, you filter an array using a boolean index list.

# A boolean index list is a list of booleans corresponding to indexes in the array.

# If the value at an index is True that element is contained in the filtered array, if the value at that index is False that element is excluded from the filtered array.

# ExampleGet your own Python Server
# Create an array from the elements on index 0 and 2:

arr_filter1 = np.array([41, 42, 43, 44])

x_filter1 = [True, False, True, False]

newarr_filter1 = arr_filter1[x_filter1]

print('newarr_filter1')
print(newarr_filter1) #[41 43]

# The example above will return [41, 43], why?

# Because the new array contains only the values where the filter array had the value True, in this case, index 0 and 2.

#######################################################
# Creating the Filter Array
# In the example above we hard-coded the True and False values, but the common use is to create a filter array based on conditions.

# Example
# Create a filter array that will return only values higher than 42:

arr_filter2 = np.array([41, 42, 43, 44])

# Create an empty list
filter_arr_filter2 = []

# go through each element in arr
for element in arr_filter2:
    # if the element is higher than 42, set the value to True, otherwise False:
    if element > 42:
        filter_arr_filter2.append(True)
    else:
        filter_arr_filter2.append(False)

newarr_filter2 = arr_filter2[filter_arr_filter2]

print('filter_arr_filter2')
print(filter_arr_filter2) #[False, False, True, True]
print('newarr_filter2')
print(newarr_filter2) #[43 44]

# Example
# Create a filter array that will return only even elements from the original array:

arr_filter3 = np.array([1, 2, 3, 4, 5, 6, 7])

# Create an empty list
filter_arr_filter3 = []

# go through each element in arr
for element in arr_filter3:
    # if the element is completely divisble by 2, set the value to True, otherwise False
    if element % 2 == 0:
        filter_arr_filter3.append(True)
    else:
        filter_arr_filter3.append(False)

newarr_filter3 = arr_filter3[filter_arr_filter3]

print('filter_arr_filter3')
print(filter_arr_filter3) #[False, True, False, True, False, True, False]
print('newarr_filter3')
print(newarr_filter3) #[2 4 6]

#######################################################
# Creating Filter Directly From Array
# The above example is quite a common task in NumPy and NumPy provides a nice way to tackle it.

# We can directly substitute the array instead of the iterable variable in our condition and it will work just as we expect it to.

# Example
# Create a filter array that will return only values higher than 42:

arr_filter4 = np.array([41, 42, 43, 44])

filter_arr_filter4 = arr_filter4 > 42

newarr_filter4 = arr_filter4[filter_arr_filter4]

print('filter_arr_filter4')
print(filter_arr_filter4) #[False False  True  True]
print('newarr_filter4')
print(newarr_filter4) #[43 44]

# Example
# Create a filter array that will return only even elements from the original array:

arr_filter5 = np.array([1, 2, 3, 4, 5, 6, 7])

filter_arr_filter5 = arr_filter5 % 2 == 0

newarr_filter5 = arr_filter5[filter_arr_filter5]

print('filter_arr_filter5')
print(filter_arr_filter5) #[False  True False  True False  True False]
print('newarr_filter5')
print(newarr_filter5) #[2 4 6]

#######################################################
#más directamente todavía:
newarr_filter6 = arr_filter5[arr_filter5 % 2 == 0]
print('newarr_filter6')
print(newarr_filter6) #[2 4 6]