# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 15:20:18 2018

@author: udayk
"""
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
            if item == letter :
                word = sentence[index_initial:index_final-1]
                index_initial = index_final
                if word != " ":
                    words.append(word)
    
    return words
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

    
            


            

            
        
        



    