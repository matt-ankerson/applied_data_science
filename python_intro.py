# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import time

# reduce will 'reduce' the iterable to a single value
factorial = lambda x: reduce(lambda x, y: x * y, range(1, x))

print (time.strftime("%I:%M:%S"))
print (time.strftime("%d/%m/%Y"))     
        
count_evens = lambda list: reduce(lambda x: x % 2 == 0, list)