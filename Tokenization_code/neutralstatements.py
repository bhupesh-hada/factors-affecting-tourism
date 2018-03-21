# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 22:39:23 2018

@author: udayk
"""



    para_sentences = ['i love cookies.','i hate cookies.',"i don't like cookies.","Why they are still here?","Are you kidding?"]
    sentences = ['love cookies.','they are still']
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
    
 print(para_sentences)   



    
    
