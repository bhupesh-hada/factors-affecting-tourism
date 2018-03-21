# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 00:32:39 2018

@author: udayk
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np






import os
os.chdir('C:/Users/udayk/Downloads')

import codecs
with codecs.open('Tamil_Nadu.csv',"r", encoding="utf-8",errors='ignore') as fdata:
    c=fdata.readlines()
    

def sentence_break(para):
  listof = [".",",","!","?"]
  sentences = []
  index_initial = 0
  index_final = 0
  while index_final < len(para):
    letter = para[index_final]
    index_final += 1
    for index,item in enumerate(listof):
      if item == letter:
        sent = para[index_initial:index_final]
        sentences.append(sent)
        index_initial = index_final
  return sentences

sentences =[]
for i in c:
    sentences.append(sentence_break(i))

s=[]     
for l in sentences:
    for j in l:
        s.append(j)

       
import codecs
with codecs.open('Collected_facility_1.csv',"r", encoding="utf-8",errors='ignore') as fdata:
    d=fdata.readlines()
    
def word_break(sentence):
    listof = [".",",","!","?"," ",'"']
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
                    words.append(word)d
    
    return words

def sentence_abstract(sentence):
    words = word_break(sentence)
    words.pop(0)
    j = 3
    words[0]=words[0][3:len(words[0])]
    sentence = ' '.join(words)
    return sentence
        
t = sentence_abstract(d[8])
print(word_break(t))       
    
    

