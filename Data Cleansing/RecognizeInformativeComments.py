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

nlp = spacy.load('de')

# Paste the file address of pickle files
files_address = str('C:/Users/Vahid/Desktop/.../...')

with open(str(file_address) + '/List_FirstName.pkl', 'rb') as f:
    List_FirstName = pickle.load(f)

with open(str(file_address) + '/List_LastName.pkl', 'rb') as f:
    List_LastName = pickle.load(f)

with open(str(file_address) + '/Dic_Comments_WordFrequency.pkl', 'rb') as f:
    Dic_Comments_WordFrequency = pickle.load(f)


with open(str(file_address) + '/Dict_Emojis.pkl', 'rb') as f:
    Dict_Emojis = pickle.load(f)

def multipleReplace(text, wordDict):
    for key in wordDict:
        text = text.replace(key, " " + str(wordDict[key])+ " ")
    return(text)

def split_upper(string):
    return(list(filter(None, re.split("([A-Z][^A-Z]*)", string))))

def cleanText(line):
    
    result = html.unescape(line)
    result = re.sub(r"\shttp\S+", "", result)
    result = re.sub(r"http\S+", "", result)
    result = re.sub(r"\swww\S+", "", result)
    result = re.sub(r"www\S+", "", result)
    result = re.sub(r"--\S+", " ", result)
    result = re.sub(r"\s--\S+", " ", result)
    result = re.sub(r"--\S+", " ", result)
    result = ''.join(''.join(s)[:3] for _, s in itertools.groupby(result))
    result = result.replace("Timeline Photos", " ").replace("No Title for this Attachment", " ").replace("No Message for this Post", " ").replace("+", " ").replace("nan", " ").replace("No Comment Message", " ").replace("Nothing", " ")
    edited_line = multipleReplace(result.replace('\n',' ').replace("@", " ").replace('\t',' ').strip(), Dict_Emojis)
    edited_line = re.sub(r'\s+', ' ', edited_line).strip()

    return(edited_line)


def FirstName_LastName_List(comment):
    
    comment_new = cleanText(comment)
    
    all_words = re.findall(r'\w+', comment_new)    
    all_words_new = all_words
    removed_words = []
    
    if all_words:
        i = -1
        for w in all_words:
            i += 1
            try:
                if str(w).isdigit() == True:
                    removed_words.append(str(w))
                elif Dic_Comments_WordFrequency[str(w).lower()] < 25:
                    removed_words.append(str(w))
                elif len(all_words) == 1 and str(w) in List_FirstName and Dic_Comments_WordFrequency[str(w).lower()] < 22000:
                    removed_words.append(str(w))
                elif str(w).lower() in ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','ä','ü','ö','ß']:
                    removed_words.append(str(w))
                elif str(w) in List_FirstName and Dic_Comments_WordFrequency[str(w).lower()] < 20000:
                    removed_words.append(str(w))
                elif str(w) in List_LastName and Dic_Comments_WordFrequency[str(w).lower()] < 20000 and len(all_words) == 2 and i != 0 and all_words[i-1] in List_FirstName:
                    removed_words.append(str(w))
                    removed_words.append(str(all_words[i-1]))
                elif split_upper(str(w)) and len(split_upper(str(w))) < 4:
                    for splitted_word in split_upper(str(w)):
                            try:
                                if str(splitted_word).lower() in ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','ä','ü','ö','ß']:
                                    removed_words.append(str(w))
                                elif str(splitted_word) in List_FirstName and Dic_Comments_WordFrequency[str(splitted_word).lower()] < 20000:
                                    removed_words.append(str(w))
                                elif str(w) in List_LastName and Dic_Comments_WordFrequency[str(splitted_word).lower()] < 20000 and len(all_words) != 1:
                                    removed_words.append(str(w))
                                else:
                                    doc = nlp(str(splitted_word))
                                    for ent in doc.ents:
                                        try:
                                            if ent.label_ == "PERSON" and Dic_Comments_WordFrequency[str(splitted_word).lower()] < 1000:
                                                removed_words.append(str(w))
                                        except:
                                            pass
                            except:
                                pass

                else:
                    doc = nlp(str(w))
                    for ent in doc.ents:
                        try:
                            if ent.label_ == "PERSON" and Dic_Comments_WordFrequency[str(w).lower()] < 1000:
                                removed_words.append(str(w))
                        except:
                            pass
            except:
                pass

        if removed_words:
            
            for W in removed_words:
                try:
                    all_words_new.remove(str(W))
                except:
                    pass
            try:
                if len(all_words_new) == 1 and Dic_Comments_WordFrequency[str(all_words_new[0]).lower()] < 1000:
                    all_words_new = []
            except:
                pass

    else:
        all_words_new = all_words

    return(all_words_new)

# Paste the file address of CSV files
file_address = str('C:/Users/Vahid/Desktop/.../...')

for filename in os.listdir(str(file_address)):

    COMMENTSTABLE = Table.read_table(str(file_address)+ "/" + str(filename), encoding="ISO-8859-1")

    ID = COMMENTSTABLE.column(0)
    PostKeyword = COMMENTSTABLE.column(1)
    PostMessage = COMMENTSTABLE.column(2)
    PostAttachment = COMMENTSTABLE.column(3)
    CommentMessage = COMMENTSTABLE.column(4)

    zipped= zip(ID, PostKeyword, PostMessage, PostAttachment, CommentMessage)

    Id = []
    P_K = []
    P_M = []
    P_A = []
    C_M = []

    not_explored_Id = []
    not_explored_P_K = []
    not_explored_P_M = []
    not_explored_P_A = []
    not_explored_C_M = []

    counter = 0
    for iD, p_k, p_m, p_a, c_m in zipped:
        counter += 1
        print(counter)
        
        all_words = re.findall(r'\w+', str(c_m))
        if len(all_words) >= 21:
            not_explored_Id.append(str(iD))
            not_explored_P_K.append(str(p_k))
            not_explored_P_M.append(str(p_m))
            not_explored_P_A.append(str(p_a))
            not_explored_C_M.append(cleanText(str(c_m)))

        elif not cleanText(str(c_m)):
            Id.append(str(iD))
            P_K.append(str(p_k))
            P_M.append(str(p_m))
            P_A.append(str(p_a))
            C_M.append(cleanText(str(c_m)))
            
        elif not FirstName_LastName_List(str(c_m)):
            Id.append(str(iD))
            P_K.append(str(p_k))
            P_M.append(str(p_m))
            P_A.append(str(p_a))
            C_M.append(cleanText(str(c_m)))
            
        else:
            not_explored_Id.append(str(iD))
            not_explored_P_K.append(str(p_k))
            not_explored_P_M.append(str(p_m))
            not_explored_P_A.append(str(p_a))
            not_explored_C_M.append(cleanText(str(c_m)))


    InformativeComments = Table().with_columns('PostCommentID', not_explored_Id,
                                               'Post Keyword', not_explored_P_K,
                                               'Post Message', not_explored_P_M,
                                               'Post Attachment', not_explored_P_A,
                                               'Comment Message', not_explored_C_M)

    InformativeComments.to_csv(str(filename) + "-InformativeComments.csv")

    UninformativeComments = Table().with_columns('PostCommentID', Id,
                                                'Post Keyword', P_K,
                                                'Post Message', P_M,
                                                'Post Attachment', P_A,
                                                'Comment Message', C_M)


    UninformativeComments.to_csv(str(filename) + "-UninformativeComments.csv")
