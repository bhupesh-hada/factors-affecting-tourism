# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 20:57:04 2018

@author: udayk
"""

def word_break(sentence):
    listof = [".",",","!","?"," "]
    one = 0
    words = []
    index_initial = 0
    index_final = 0 
    while index_final < len(sentence):
        letter = sentence[index_final]
        index_final += 1
        for index,item in enumerate(listof):
            if item == letter:
                word = sentence[index_initial:index_final-1]
                index_initial = index_final
                if word != " ":
                    words.append(word)
    
    return words




            