# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 18:11:42 2018

@author: bhupe
"""
para = ['i love cookies.','i hate cookies.',"i don't like cookies.","Why they are still here?","Are you kidding?"]
sentences = ['love cookies.','they are still']
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
def neutral_sentences(para,sentences):
    para_sentences = para
    neutral = []
    statements = []
    for sentence in sentences:
        sentense_list = sentence_break(sentence)
        for llsentence in sentense_list:
            statements.append(llsentence)
    for i in range(0,len(para_sentences)):
        psentence_words = word_break(para_sentences[i])
        for statement in statements:
            statement_word = word_break(statement)
            for index1,item1 in enumerate(statement_word):
                for index2,item2 in enumerate(psentence_words):
                    if item1 == item2:
                        statement_word.pop(index1)
            if len(statement_word) <= 1:
                neutral.append(i)
         
    for num in neutral:
        para_sentences.pop(num)
        for i in range(0,len(neutral)):
            neutral[i] = neutral[i]-1
    return (para_sentences)
         


print(para_sentences(para,sentences))