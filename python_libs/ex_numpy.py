# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:26:51 2015

@author: matt
"""

from numpy import *

# Two dimensional array, courtesy of numpy
a = arange(15).reshape(3, 5)
# output entire array
print(a)
# output the dimensions of the array (rank)
print(a.shape)
# output the number of axis (dimensions)
print(a.ndim)
# descriptor of the objects inside the array
print(a.dtype.name)
# the size in bytes of each element of the array
print(a.itemsize)
# the total number of elements in the array (essentially a product of dimensions)
print(a.size)
# the type of this array
print(type(a))


# Array Creation
a = array( [2,3,4] )
# output array
print(a)
b = array( [1.2, 3.5, 5.1] )
print(b.dtype)
# array transforms sequences of sequences into two dimensional arrays
b = array([(1.5, 2, 3),(4, 5, 6)])
print(b)
# explicitly define array type
c = array([[1, 2], [3, 4]], dtype=complex)
print(c)
# create an array with empty placeholder content
p = zeros((3, 5))
print(p)