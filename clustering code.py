# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 23:03:32 2018

@author: bhupe
"""

import pandas as pd
import os
os.chdir('C:/Users/udayk/Desktop/tourism')

a=[]
b=[]
import codecs
with codecs.open('environment_Tamil_Nadu_3.csv', "r",encoding='utf-8', errors='ignore') as fdata:
    c=fdata.readlines()
    for i in range(0,len(c)):
        
        d=c[i][len(c[i])-3]
        b.append(d)
        #while(c[e-1]=='\t'):
        #    e=e-1
        f=len(c[i])-4
        a.append(c[i][0:f])
        
        
dataset=pd.DataFrame({'Reviews':a,'Ratings':b},columns=['Reviews','Ratings'])  

p = 0
n = 0
for i in range(0,len(b)):
    if b[i] == '0':
        n+=1
    if b[i] == '1':
        p+=1
"""print(p)
print(n)
per = (p/(p+n))*100
print(per)"""

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
            if item == letter:
                word = sentence[index_initial:index_final-1]
                index_initial = index_final
                if word != "":
                    words.append(word)
    
    return words



s=a
df=pd.DataFrame(data=s)    
import re
import nltk
#nltk.download('stopwords')
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
    
    
    
    
# =============================================================================
# #hospitality_factor
# clean_df=pd.read_csv('clean_factor.csv')
# # applying stemming
# import re
# import nltk
# nltk.download('stopwords')
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# corpus_clean = []
# for i in range(0, len(clean_df)):
#     review = re.sub('[^a-zA-Z]', ' ',clean_df['cleanliness'][i])
#     review = review.lower()
#     review = review.split()
#     ps = PorterStemmer()
#     review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
#     review = ' '.join(review)
#     corpus_clean.append(review)
# =============================================================================
                              
    
                             
                              
    
# =============================================================================
#     
# final_list_clean1_Rajasthan=[]
# index_clean=[]
# for i in range(0,len(s)):
#     wr_br=word_break(corpus_1[i])
#     for j in range(len(corpus_clean)):
#         if corpus_clean[j] in wr_br:
#             index_clean.append(i)
#             statement=' '.join(wr_br)
#             final_list_clean1_Rajasthan.append(statement)
#             break 
#         
#         
# actual_statement_clean_Rajasthan=[]
# for i in index_clean:
#     actual_statement_clean_Rajasthan.append(df[0][i])        
# 
# =============================================================================



index = []
positive=[]
negative=[]


word_list_con2 = ['delay','wait','tired','journey','comfort','uncmofort'] 
word_list_con3 = ['bump','road','bus']
word_list_con4 = ['train','station']
word_list_cle1 = ['toilet']

word_list_cle3 = ['Rubbish','dustbin','bin','Trash','Garbage']
word_list_cle4 = ['cockroaches','rat','urine','litter']
word_list_cle5 = ['smells']
word_list_cle6 = ['manintain','organize','maintainance']
word_list_fac1 = ['atm','cash','money']
word_list_fac2 = ['power','powercut','electricity']
word_list_fac3 = ['charge','costly','expensive','cheap','shop']
word_list_fac4 = ['queue']
word_list_fac5 = ['internet','wifi','network']
word_list_fac6 = ['geysers','heaters','air condition','comfort','spatious','ultra modern','spatious','luxury','pool','functional','stay','decent','AC','housekeeping','accomodation']
word_list_env1 = ['horn','pollution','populated','chaos','chaotic','traffic','jammed','crowd']

word_list= word_list_env1
for k in word_list:
    index = []
    review = re.sub('[^a-zA-Z]', ' ',k)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    m=review
    
    
    for i in range(0,len(a)):
        
        listo = word_break(corpus_1[i])
        
        if m in listo:
            index.append(i)
    pos = 0
    neg = 0
    for j in index:
        if b[j] == '0':
            neg += 1
        if b[j] == '1':
            pos += 1
    #print(pos)
    #print(neg)
    positive.append(pos)
    negative.append(neg)
print(sum(positive))
print(sum(negative))
      