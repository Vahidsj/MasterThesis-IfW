import spacy
from datascience import *
import html
import re
import itertools
import string
import os
import numpy as np
from ast import literal_eval as make_list
import operator
from collections import Counter
import pickle
import gensim

nlp = spacy.load('de')
model = gensim.models.Word2Vec.load('C:/Users/Vahid/Desktop/MODEL-WORD2VEC/Size300.model')

with open('C:/Users/Vahid/Desktop/Dic_Comments_WordFrequency.pkl', 'rb') as f:
    Dic_Comments_WordFrequency = pickle.load(f)

def word_frequency(word):
    try:
        return(int(Dic_Comments_WordFrequency[str(word)]))
    except:
        pass

with open('C:/Users/Vahid/Desktop/Dict_Error_Suggestion.pkl', 'rb') as f:
    Dict_Error_Suggestion = pickle.load(f)

def error_suggestion(word):
    try:
        return(Dict_Error_Suggestion[str(word)])
    except:
        pass

with open('C:/Users/Vahid/Desktop/Dic_Error_Suggestion_Split.pkl', 'rb') as f:
    Dic_Error_Suggestion_Split = pickle.load(f)

def splits_two_words(word):
    try:
        return(Dic_Error_Suggestion_Split[str(word)])
    except:
        pass

with open('C:/Users/Vahid/Desktop/Dic_Error_Suggestion_One_Edit.pkl', 'rb') as f:
    Dic_Error_Suggestion_One_Edit = pickle.load(f)

def find_suggested_words_one_edit(word):
    try:
        return(Dic_Error_Suggestion_One_Edit[str(word)])
    except:
        pass

with open('C:/Users/Vahid/Desktop/Dic_Error_Suggestion_Repeated_Letters.pkl', 'rb') as f:
    Dic_Error_Suggestion_Repeated_Letters = pickle.load(f)

def find_suggested_words_repeated_letters(word):
    try:
        return(Dic_Error_Suggestion_Repeated_Letters[str(word)])
    except:
        pass

with open('C:/Users/Vahid/Desktop/List_FirstName.pkl', 'rb') as f:
    List_FirstName = pickle.load(f)

with open('C:/Users/Vahid/Desktop/List_LastName.pkl', 'rb') as f:
    List_LastName = pickle.load(f)


COMMENTSTABLE = Table.read_table("C:/Users/Vahid/Desktop/IncorrectComments.csv", encoding="ISO-8859-1")

ID = COMMENTSTABLE.column(0)
Comments = COMMENTSTABLE.column(1)
Errors = []

for i in COMMENTSTABLE.column(2):
    Errors.append(make_list(i))


def spacy_ent_check(word):

    docs_list = [str(word), str(word.lower()), str(word.capitalize())]

    for item in docs_list:
        doc = nlp(str(item))
        for ent in doc.ents:
            if ent.label_:
                return(item)
                break

def spacy_tag_check(word):

    docs_list = [str(word), str(word.lower())]

    for item in docs_list:
        doc = nlp(str(item))
        for word in doc:
            if word.tag_ and word.tag_ not in ['VVFIN', 'NN', 'XY']:
                return(item)
                break


def FirstName_LastName_List(word, comment):
    
    docs_list = [str(word), str(word.lower()), str(word.capitalize())]
    for item in docs_list:
        if str(item) in List_FirstName:
            return(item)
            break
        elif str(item) in List_LastName:
            return(item)
            break

    list_of_words = re.findall(r'\w+', comment)
    try:
        word_index = list_of_words.index(str(word))
        if word_index != 0:
            doc = nlp(str(list_of_words[word_index - 1]))
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    return(word)

        elif word_index != len(list_of_words) - 1:
            doc = nlp(str(list_of_words[word_index + 1]))
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    return(word)
    except:
        pass

corrected_words = {}
zipped_0 = zip(ID, Comments, Errors)

Corrected_CommentID = []
Corrected_Comment = []
NotChanged_CommentID = []
NotChanged_Comment = []
for i,c,e in zipped_0:

    comment = str(c)
    checker = []
    for j in e:
        
        if FirstName_LastName_List(str(j), comment):
            comment = comment.replace(str(j), FirstName_LastName_List(str(j), comment))
            if str(j) not in list(corrected_words.keys()):
                corrected_words[str(j)] = FirstName_LastName_List(str(j), comment)
            checker.append(1)
                
        elif spacy_ent_check(str(j)):
            comment = comment.replace(str(j), spacy_ent_check(str(j)))
            if str(j) not in list(corrected_words.keys()):
                corrected_words[str(j)] = spacy_ent_check(str(j))
            checker.append(1)
            
        elif find_suggested_words_one_edit(str(j)) and len(find_suggested_words_one_edit(str(j))) == 1:
            comment = comment.replace(str(j), find_suggested_words_one_edit(str(j))[0])
            if str(j) not in list(corrected_words.keys()):
                corrected_words[str(j)] = find_suggested_words_one_edit(str(j))[0]
            checker.append(1)
            
        elif find_suggested_words_repeated_letters(str(j)):
            number = 0
            for r_l in find_suggested_words_repeated_letters(str(j)):
                if word_frequency(str(r_l).lower()):
                    if word_frequency(str(r_l).lower()) > number:
                        replacement_word_repeated_letters = str(r_l)
                        number = word_frequency(str(r_l).lower())
                    comment = comment.replace(str(j), str(replacement_word_repeated_letters))
            if str(j) not in list(corrected_words.keys()):
                corrected_words[str(j)] = str(replacement_word_repeated_letters)
            checker.append(1)

        elif splits_two_words(str(j)) and not find_suggested_words_one_edit(str(j)):
            comment = comment.replace(str(j), (splits_two_words(str(j))[0] + " " + splits_two_words(str(j))[1]))
            if str(j) not in list(corrected_words.keys()):
                corrected_words[str(j)] = (splits_two_words(str(j))[0] + " " + splits_two_words(str(j))[1])
            checker.append(1)

        elif find_suggested_words_one_edit(str(j)) and len(find_suggested_words_one_edit(str(j))) < 6:
            number = 0
            for one_e in find_suggested_words_one_edit(str(j)):
                if word_frequency(str(one_e).lower()):
                    if word_frequency(str(one_e).lower()) > number:
                        replacement_word_one_edit = str(one_e)
                        number = word_frequency(str(one_e).lower())
                    comment = comment.replace(str(j), replacement_word_one_edit)
                if str(j) not in list(corrected_words.keys()):
                    corrected_words[str(j)] = str(replacement_word_one_edit)
            checker.append(1)

        else:
            print(j)

    print(len(checker) == len(e), e, checker)
    
    if len(checker) == len(e):

        Corrected_CommentID.append(str(i))
        Corrected_Comment.append(str(comment))

    else:

        NotChanged_CommentID.append(str(i))
        NotChanged_Comment.append(str(comment))


post_comment_corrected = Table().with_columns('PostCommentID', Corrected_CommentID,
                                              'Comments', Corrected_Comment)

post_comment_corrected.to_csv("CorrectedComments.csv")

post_comment_notchanged = Table().with_columns('PostCommentID', NotChanged_CommentID,
                                               'Comments', NotChanged_Comment)

post_comment_notchanged.to_csv("Comments_NoChange.csv")
            
