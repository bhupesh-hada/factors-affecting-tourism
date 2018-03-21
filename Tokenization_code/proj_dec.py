# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 22:05:12 2018

@author: bhupe
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
os.chdir('C:/Users/bhupe/Desktop/Factors affecting Tourism/Factors/Facility')




a=[]
b=[]
import codecs
with codecs.open('facility_final.txt', "r",encoding='utf-8', errors='ignore') as fdata:
    c=fdata.readlines()
     for i in range(1,len(c)):
        
        d=c[i][len(c[i])-3]
        b.append(d)
        #while(c[e-1]=='\t'):
        #    e=e-1
        f=len(c[i])-14
        a.append(c[i][0:f])
        
        
        
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




a1=[]
a2=[]
a3=[]
for item in a:
    if len(item)!=0:
        a1.append(item)
listof = [".",",","!","?"," ",'"']
for item in a1:
    if not item[len(item)-1] in listof:
        item = item+"."
    lut = sentence_break(item)
    for stu in lut:
        a2.append(stu)
for jin in a2:
    if len(jin) >= 3:
        a3.append(jin)
        
        
        
        
        
        
        
        
import codecs
with codecs.open('Blogs.csv',"r", encoding="utf-8",errors='ignore') as fdata:
    x=fdata.readlines()        

        
        
sentences =[]
for i in x:
    sentences.append(sentence_break(i))

s=[]     
for l in sentences:
    for j in l:
        s.append(j) 

        
        
def word_break(sentence):
    listof = [".",",","!","?"," ",'"',"(",")"]
    one = 0
    words = []
    index_initial = 0
    index_final = 0 
    while index_final < len(sentence):
        letter = sentence[index_final]
        index_final += 1
        for index,item in enumerate(listof):
            if item == letter :
                word = sentence[index_initial:index_final-1]
                index_initial = index_final
                if word != "":
                    words.append(word)
    
    return words




def neutral_sentences(total,sample):
    index = []
    neutral = []
    for i in range(0,len(total)):
        word_total = word_break(total[i])
        for sentence in sample:
            word_sentence = word_break(sentence)
            length = len(word_sentence)
            for word_s in word_sentence:
                for word_t in word_total:
                    if word_s == word_t:
                        length -= 1
            if length == 0:
                index.append(i)
    for j in range(0,len(total)):
        if not j in index:
            neutral.append(total[j])
    return neutral
            

    
    
s1 = []
s2 = []
for item in s:
    if len(item)!=0:
        s1.append(item)
for jin in s1:
    if len(jin) >= 4:
        s2.append(jin)    

neutrals = neutral_sentences(s2,a3)    


df=pd.DataFrame({'Reviews':a,'Ratings':b},columns=['Reviews','Ratings'])        





import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,len(df)):
    review = re.sub('[^a-zA-Z]', ' ',str(df['Reviews'][i]))
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
    
    
    
    
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus_1 = []
for i in range(0,len(df)):
    review = re.sub('[^a-zA-Z]', ' ',str(neutrals[i]))
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus_1.append(review)    
 
relevant=[]   
for i in corpus_1:
    xyz=i.split()
    for j in corpus:
        num=0
        xyz_1=j.split()
        for k in xyz:
            if k in xyz_1:
                num=num+1
                xyz_1.remove(k)
            if num>2:
                relevant.append(i)
                break
            
        
        
        
        
        
                