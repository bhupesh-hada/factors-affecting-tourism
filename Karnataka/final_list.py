# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 09:18:58 2018

@author: bhupe
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
os.chdir('C:/Users/bhupe/Desktop/Factors affecting Tourism/Karnataka')




a=[]
b=[]
import codecs
with codecs.open('Karnataka.csv', "r",encoding='utf-8', errors='ignore') as fdata:
    c=fdata.readlines()
    
    
    
    
with codecs.open('Karnataka_1.csv', "r",encoding='utf-8', errors='ignore') as fdata:
    x=fdata.readlines() 
    
    
    
join=c+x  

#importing the csv file containing relevant words
environ_df=pd.read_csv('environ_factor.csv')
# applying stemming
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, len(environ_df)):
    review = re.sub('[^a-zA-Z]', ' ', environ_df['environment'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
    
    
    
    
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
for i in join:
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
    
df=pd.DataFrame(data=s)    
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus_1 = []
for i in range(0, len(df)):
    review = re.sub('[^a-zA-Z]', ' ',df[0][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus_1.append(review)    
    

final_list_environ=[]
index_environ=[]
for i in range(0,len(s)):
    wr_br=word_break(corpus_1[i])
    for j in range(len(corpus)):
        if corpus[j] in wr_br:
            statement=' '.join(wr_br)
            index_environ.append(i)
            final_list_environ.append(statement)
            break
        
actual_statement_environ=[]
for i in index_environ:
    actual_statement_environ.append(df[0][i])
        
        
            
# clean_factor       
clean_df=pd.read_csv('clean_factor.csv')
# applying stemming
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus_clean = []
for i in range(0, len(clean_df)):
    review = re.sub('[^a-zA-Z]', ' ',clean_df['cleanliness'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus_clean.append(review)    


final_list_clean=[]
index_clean=[]
for i in range(0,len(s)):
    wr_br=word_break(corpus_1[i])
    for j in range(len(corpus_clean)):
        if corpus_clean[j] in wr_br:
            index_clean.append(i)
            statement=' '.join(wr_br)
            final_list_clean.append(statement)
            break

actual_statement_clean=[]
for i in index_clean:
    actual_statement_clean.append(df[0][i])        
        

#conveyance_factor
conveyance_df=pd.read_csv('conveyance_factor.csv')
# applying stemming
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus_conveyance = []
for i in range(0, len(conveyance_df)):
    review = re.sub('[^a-zA-Z]', ' ',conveyance_df['conveyance'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus_conveyance.append(review)   
    
final_list_conveyance=[]
index_conveyance=[]
for i in range(0,len(s)):
    wr_br=word_break(corpus_1[i])
    for j in range(len(corpus_conveyance)):
        if corpus_conveyance[j] in wr_br:
            index_conveyance.append(i)
            statement=' '.join(wr_br)
            final_list_conveyance.append(statement)
            break 
        
actual_statement_conveyance=[]
for i in index_conveyance:
    actual_statement_conveyance.append(df[0][i])        
        
    
        
        
#food_factor
food_df=pd.read_csv('food_factor.csv')
# applying stemming
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus_food = []
for i in range(0, len(food_df)):
    review = re.sub('[^a-zA-Z]', ' ',food_df['food'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus_food.append(review)   
    
final_list_food=[]
index_food=[]
for i in range(0,len(s)):
    wr_br=word_break(corpus_1[i])
    for j in range(len(corpus_food)):
        if corpus_food[j] in wr_br:
            index_food.append(i)
            statement=' '.join(wr_br)
            final_list_food.append(statement)
            break 
        
        
actual_statement_food=[]
for i in index_food:
    actual_statement_food.append(df[0][i])        



#hospitality_factor
hospitality_df=pd.read_csv('hospitality_factor.csv')
# applying stemming
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus_hospitality = []
for i in range(0, len(hospitality_df)):
    review = re.sub('[^a-zA-Z]', ' ',hospitality_df['hospitality'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus_hospitality.append(review)   

corpus_hospitality.pop(16)
corpus_hospitality.pop(16)
corpus_hospitality.pop(21)    
    
final_list_hospitality=[]
index_hospitality=[]
for i in range(0,len(s)):
    wr_br=word_break(corpus_1[i])
    for j in range(len(corpus_hospitality)):
        if corpus_hospitality[j] in wr_br:
            index_hospitality.append(i)
            statement=' '.join(wr_br)
            final_list_hospitality.append(statement)
            break 
        
actual_statement_hospitality=[]
for i in index_hospitality:
    actual_statement_hospitality.append(df[0][i])        
         
         