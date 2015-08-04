# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 12:58:59 2015

@author: matt
"""

class Student:
    # A class representing a student
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def get_age(self):
        return self.age
        
    def set_age(self, new_age):
        self.age = new_age
        
class CompSciStudent(Student):
    ''' A class extending Student '''
    def __init__(self, name, age, section_num):
        Student.__init__(self, name, age)   # call parent's constructor
        self.section_num = section_num
        
    def get_pretty_age(self):
        return "Age: " + str(self.age)
        
class Teacher:
    # A class representing a Teacher
    def __init__(self, name):
        self.name = name
    
    def print_name(self):
        print self.name
        
class Sample:
    x = 23
    def __init__(self):
        self.__class__.x += 1   # increment the instance counter.
        
class Counter:
    overall_total = 0   # class attribute
    
    def __init__(self):
        self.my_total = 0
        
    def increment(self):
        self.__class__.overall_total += 1
        self.my_total += 1
        
class Atom:
    """ A class representing an atom """
    def __init__(self, atno, x, y, z):
        self.atno = atno
        self.position = (x, y, z)
    
    def symbol(self):
        return Atno_to_Symbol[atno]
        
    def __repr__(self):
        return '%d %10.4f %10.4f %10.4f' % (self.atno, self.position[0], self.position[1], self.position[2])
        
class Library(object):
    def __init__(self):
        self.books = ["Book 1", "Book 2", "Book 3"]
    
    def __getitem__(self, i):
        return self.books[i]
        