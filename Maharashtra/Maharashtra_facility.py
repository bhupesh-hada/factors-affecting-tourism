# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 11:07:24 2018

@author: bhupe
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
os.chdir('C:/Users/bhupe/Desktop/Factors affecting Tourism/Karnataka_+_Uttar_Pradesh')




a=[]
b=[]
import codecs
with codecs.open('actual_statement_facility.txt', "r",encoding='utf-8', errors='ignore') as fdata:
    c=fdata.readlines()
    for i in range(0,len(c)):
        
        d=c[i][len(c[i])-3]
        b.append(d)
        #while(c[e-1]=='\t'):
        #    e=e-1
        f=len(c[i])-4
        a.append(c[i][0:f])
        
        
dataset=pd.DataFrame({'Reviews':a,'Ratings':b},columns=['Reviews','Ratings'])  


# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Reviews'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) 

#importing the Blogs
import os
os.chdir('C:/Users/bhupe/Desktop/Factors affecting Tourism/Maharashtra')




import codecs
with codecs.open('Maharashtra.csv', "r",encoding='utf-8', errors='ignore') as fdata:
    c1=fdata.readlines()
    
    
    
    
with codecs.open('Maharashtra_1.csv', "r",encoding='utf-8', errors='ignore') as fdata:
    c2=fdata.readlines() 
    
    
join=c1+c2


#preprocessing the Blogs
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
    
    
    
#hospitality_factor
facility_df=pd.read_csv('facility_factor.csv')
# applying stemming
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus_facility = []
for i in range(0, len(facility_df)):
    review = re.sub('[^a-zA-Z]', ' ',facility_df['facility'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus_facility.append(review)
    
final_list_facility=[]
index_facility=[]
for i in range(0,len(s)):
    wr_br=word_break(corpus_1[i])
    for j in range(len(corpus_facility)):
        if corpus_facility[j] in wr_br:
            index_facility.append(i)
            statement=' '.join(wr_br)
            final_list_facility.append(statement)
            break 
        
        
actual_statement_facility=[]
for i in index_facility:
    actual_statement_facility.append(df[0][i])  
    
    
abc=corpus+final_list_facility
 


# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(abc).toarray()
y_train1 = dataset.iloc[:, 1].values


X_train1=X[0:2693]
X_test1=X[2693:]



# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train1, y_train1)
#==============================================================================
# 
# import os
# mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-5.3.0-posix-seh-rt_v4-rev0\\mingw64\\bin'
#  
# os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
# 
# from xgboost import XGBClassifier
# classifier=XGBClassifier()
# classifier.fit(X_train1,y_train1)
#==============================================================================




# Predicting the Test set results
y_pred = classifier.predict(X_test1)


# Predicting the Test set results
#y_pred = classifier.predict(X_test1)

df_final=pd.DataFrame({'Review':actual_statement_facility,'Rating':y_pred},columns=['Review','Rating'])
df_final.to_csv('facility_Maharashtra_3.csv')
      
                              
        
    
        
