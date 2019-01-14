# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 10:41:07 2019

@author: Zymieth
"""
with open('L:\Algorithms\kargerMinCut.txt') as file:
    rows = [n for n in [line.split('\t')[:-1] for line in file]]
    #rows = [[n for n in line.split(',')] for line in file]

def rand_contraction(graph):
    contraction_list = []fw
    
    