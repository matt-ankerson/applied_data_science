# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:32:27 2015

@author: matt
"""

class Horse():
    
    def __init__(self):
        self.name = ""
        
    def get_name_from_console(self):
        self.name = input('Enter the name of this horse.')
        
    def print_pretty_name(self):
        print 'The name of this horse is ' + self.name.upper()
        
class Circle():
    pi = 3.14
    def __init__(self, radius):
        self.radius = radius
        
    def compute_area(self):
        return (self.__class__.pi * (self.radius * self.radius))
        
