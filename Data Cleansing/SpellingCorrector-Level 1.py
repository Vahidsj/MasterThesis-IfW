import enchant
from enchant.checker import SpellChecker
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

chkr = SpellChecker("de_DE")
de = enchant.Dict("de_DE")
en = enchant.Dict("en_EN")

with open('Dic_Error_Suggestion_Split.pkl', 'rb') as f:
    Dic_Error_Suggestion_Split = pickle.load(f)

with open('Dic_Error_Suggestion_One_Edit.pkl', 'rb') as f:
    Dic_Error_Suggestion_One_Edit = pickle.load(f)

with open('Dic_Error_Suggestion_Repeated_Letters.pkl', 'rb') as f:
    Dic_Error_Suggestion_Repeated_Letters = pickle.load(f)

with open('Dict_Error_Suggestion.pkl', 'rb') as f:
    Dict_Error_Suggestion = pickle.load(f)

IRRELEVANTCOMMENTSTABLE = Table.read_table("C:/Users/Vahid/Desktop/....csv", encoding="ISO-8859-1")
IDS = IRRELEVANTCOMMENTSTABLE.column(2)
IRRELEVANT = IRRELEVANTCOMMENTSTABLE.column(6)

zipped_00 = zip(IDS, IRRELEVANT)
Irrelevant_ID = []
for i,j in zipped_00:
    if j == 1:
        Irrelevant_ID.append(i)


COMMENTSTABLE = Table.read_table("C:/Users/Vahid/Desktop/....csv", encoding="ISO-8859-1")

ID_1 = COMMENTSTABLE.column(0)
Comments_1 = COMMENTSTABLE.column(1)

ID = []
Comments = []
zipped_0 = zip(ID_1, Comments_1)
for i, j in zipped_0:
    if i not in ID and i not in Irrelevant_ID:
        ID.append(str(i))
        Comments.append(str(j))

post_comment_without_repeat = Table().with_columns('PostCommentID', ID,
                                                   'Comments', Comments)

post_comment_without_repeat.to_csv("TrainingSetComments-WithoutRepeat.csv")


comments = []
commentID = []

SpecialCharachterDict = {'ae':'ä', 'oe':'ö', 'ue':'ü', 'Ae':'Ä', 'Oe':'Ö', 'Ue':'Ü'}

zipped = zip(ID, Comments)

for Id, text in zipped:

    #Escaping HTML characters
    result = html.unescape(text)
    #Eliminate URL Link
    result = re.sub(r"\shttp\S+", "", result)
    result = re.sub(r"http\S+", "", result)
    result = re.sub(r"\swww\S+", "", result)
    result = re.sub(r"www\S+", "", result)
    #Eliminate NewLine and specific charachters
    result = result.replace("\n", " ").replace("\r", " ").replace('"', " ").replace("*", "").replace("--", "").replace("'s", " es").strip()
    result = re.sub(r"--\S+", "", result)
    result = re.sub(r"\s--\S+", "", result)
    #Standardizing words
    result = ''.join(''.join(s)[:3] for _, s in itertools.groupby(result))
    #Eliminate duplicate whitespaces
    result = re.sub(r'\s+', " ", result)
    #Replace ß >> Regarding Upper and Lower case
    list_of_words = re.findall(r'\w+', result)

    for w in list_of_words:
        
        if "ß" in str(w):
            if str(w).replace("ß", "").isupper() == True:
                edited_w = str(w).replace("ß", "SS")
                result = result.replace(str(w), str(edited_w))

        if "ae" in str(w) and np.logical_and(np.logical_and(de.check(str(w)) == False, de.check(str(w).capitalize()) == False) , np.logical_and(en.check(str(w)) == False, en.check(str(w).lower()) == False)):
                edited_w = str(w).replace("ae", "ä")
                result = result.replace(str(w), str(edited_w))

        if "oe" in str(w) and np.logical_and(np.logical_and(de.check(str(w)) == False, de.check(str(w).capitalize()) == False) , np.logical_and(en.check(str(w)) == False, en.check(str(w).lower()) == False)):
                edited_w = str(w).replace("oe", "ö")
                result = result.replace(str(w), str(edited_w))


        if "ue" in str(w) and np.logical_and(np.logical_and(de.check(str(w)) == False, de.check(str(w).capitalize()) == False) , np.logical_and(en.check(str(w)) == False, en.check(str(w).lower()) == False)):
                edited_w = str(w).replace("ue", "ü")
                result = result.replace(str(w), str(edited_w))


        if np.logical_and(de.check(str(w)) == False, de.check(str(w).capitalize()) == False) and np.logical_or(np.logical_and("Ae" in str(w), en.check(str(w).lower()) == False), np.logical_and("AE" in str(w), en.check(str(w).lower()) == False)):
                edited_w = str(w).replace("Ae", "Ä").replace("AE", "Ä")
                result = result.replace(str(w), str(edited_w))


        if np.logical_and(de.check(str(w)) == False, de.check(str(w).capitalize()) == False) and np.logical_or(np.logical_and("Oe" in str(w), en.check(str(w).lower()) == False), np.logical_and("OE" in str(w), en.check(str(w).lower()) == False)):
                edited_w = str(w).replace("Oe", "Ö").replace("OE", "Ö")
                result = result.replace(str(w), str(edited_w))


        if np.logical_and(de.check(str(w)) == False, de.check(str(w).capitalize()) == False) and np.logical_or(np.logical_and("Ue" in str(w), en.check(str(w).lower()) == False), np.logical_and("UE" in str(w), en.check(str(w).lower()) == False)):
                edited_w = str(w).replace("Ue", "Ü").replace("UE", "Ü")
                result = result.replace(str(w), str(edited_w))

    comments.append(str(result))
    commentID.append(str(Id))



POSTCOMMENTID = []
COMMENTS = []
ERROR_WORDS = []
SUGGEST_WORDS = []
ERROR_WORDS_LIST = []
SUGGEST_WORDS_LIST = []

zipped_1 = zip(commentID, comments)

for Id, text in zipped_1:
    
    chkr.set_text(text)
    
    for error in chkr:
        
        ERROR_WORDS.append(str(error.word))
        SUGGEST_WORDS.append(str(error.suggest()))
        print(str(error.word))

    if ERROR_WORDS:

        POSTCOMMENTID.append(str(Id))
        COMMENTS.append(str(chkr.get_text()))
        ERROR_WORDS_LIST.append(ERROR_WORDS)
        SUGGEST_WORDS_LIST.append(SUGGEST_WORDS)
        ERROR_WORDS = []
        SUGGEST_WORDS = []


def find_suggested_words_repeated_letters(word):

    repeated_letters = []
    for i in list(''.join(s) for _, s in itertools.groupby(word)):
        if len(i) >= 2:
            repeated_letters.append(str(i))

    main_list = []
    combination_list = []
    for k in repeated_letters:
        for n in range(1,len(k)+1):
            combination_list.append(k[:n])
        main_list.append(combination_list)
        combination_list = []
    
    all_combination_list = list(itertools.product(*main_list))
    list_all_combination_list = list(map(list, all_combination_list))

    combination_repeated_letters_and_combination_list = []
    for i in list_all_combination_list:
        for j in [repeated_letters]:
            combination_repeated_letters_and_combination_list.append([j,i])

    def multiple_replacement(text,rep):

        try:
            rep = dict((re.escape(k), v) for k, v in rep.items())
            pattern = re.compile("|".join(rep.keys()))
            text = pattern.sub(lambda m: rep[re.escape(m.group(0))], str(text))
        except:
            pass

        return (text)

    related_words = []
    for i in combination_repeated_letters_and_combination_list:
        
        if de.check(str(multiple_replacement(word, dict(zip(i[0],i[1]))))) == True:
            related_words.append(multiple_replacement(word, dict(zip(i[0],i[1]))))

        elif de.check(str(multiple_replacement(word, dict(zip(i[0],i[1])))).capitalize()) == True:
            related_words.append(multiple_replacement(word, dict(zip(i[0],i[1]))))

        elif de.check(str(multiple_replacement(word, dict(zip(i[0],i[1])))).lower()) == True:
            related_words.append(multiple_replacement(word, dict(zip(i[0],i[1]))))

        elif de.check(str(multiple_replacement(word, dict(zip(i[0],i[1])))).upper()) == True:
            related_words.append(multiple_replacement(word, dict(zip(i[0],i[1]))))

    if len(related_words) > 1:
        C = list(Counter(related_words).keys())
        related_words = C

    Related_words = []
    upper_letters = []
    lower_letters = []
    
    for j in  related_words:

        for i in list(str(j))[1:]:

            if str(i).isupper() == True:

                upper_letters.append(str(i))

            else:

                lower_letters.append(str(i))


        if len(upper_letters) >= len(lower_letters):

            Related_words.append(str(j).upper())

        else:

            if j[0].isupper() == True:

                Related_words.append(str(j).capitalize())

            else:

                Related_words.append(str(j).lower())

    if not Related_words:
        
        WORD = word.lower()
        repeated_letters = []
        for i in list(''.join(s) for _, s in itertools.groupby(WORD)):
            if len(i) >= 2:
                repeated_letters.append(str(i))

        main_list = []
        combination_list = []
        for k in repeated_letters:
            for n in range(1,len(k)+1):
                combination_list.append(k[:n])
            main_list.append(combination_list)
            combination_list = []
        
        all_combination_list = list(itertools.product(*main_list))
        list_all_combination_list = list(map(list, all_combination_list))

        combination_repeated_letters_and_combination_list = []
        for i in list_all_combination_list:
            for j in [repeated_letters]:
                combination_repeated_letters_and_combination_list.append([j,i])

        def multiple_replacement(text,rep):

            try:
                rep = dict((re.escape(k), v) for k, v in rep.items())
                pattern = re.compile("|".join(rep.keys()))
                text = pattern.sub(lambda m: rep[re.escape(m.group(0))], str(text))
            except:
                pass

            return (text)

        Related_words = []
        for i in combination_repeated_letters_and_combination_list:
            
            if de.check(str(multiple_replacement(WORD, dict(zip(i[0],i[1]))))) == True:
                Related_words.append(multiple_replacement(WORD, dict(zip(i[0],i[1]))))

            elif de.check(str(multiple_replacement(WORD, dict(zip(i[0],i[1])))).capitalize()) == True:
                Related_words.append(multiple_replacement(WORD, dict(zip(i[0],i[1]))))

            elif de.check(str(multiple_replacement(WORD, dict(zip(i[0],i[1])))).lower()) == True:
                Related_words.append(multiple_replacement(WORD, dict(zip(i[0],i[1]))))

            elif de.check(str(multiple_replacement(WORD, dict(zip(i[0],i[1])))).upper()) == True:
                Related_words.append(multiple_replacement(WORD, dict(zip(i[0],i[1]))))
                
        
    return(Related_words)

def find_suggested_words_one_edit(word):

    suggested_word = []

    for i in nocheck_find_suggested_words(word):
        
        for j in edits1(str(i)):
            
            if de.check(str(j)) == True:
                suggested_word.append(str(j))

            elif de.check(str(j).capitalize()) == True:
                suggested_word.append(str(j).capitalize())

            elif de.check(str(j).upper()) == True:
                suggested_word.append(str(j).upper())
        
    return(suggested_word)


def splits_two_words(text, start=1, L=40):
    suggested_word = []
    "Return a list of all (first, rest) pairs; start <= len(first) <= L."
    for i in range(start, min(len(text), L)):
        if de.check(str(text[:i])) == True and de.check(str(text[i:])) == True and len(str(text[:i])) > 1 and len(str(text[i:])) > 1:
            suggested_word.append(str(text[:i]))
            suggested_word.append(str(text[i:]))
            
    return(suggested_word)

#################################################################################################################################################
#################################################################################################################################################

def edits1(word):
    "Return all strings that are one edit away from this word."
    pairs      = splits(word)
    deletes    = [a+b[1:]           for (a, b) in pairs if b]
    transposes = [a+b[1]+b[0]+b[2:] for (a, b) in pairs if len(b) > 1]
    replaces   = [a+c+b[1:]         for (a, b) in pairs for c in alphabet if b]
    inserts    = [a+c+b             for (a, b) in pairs for c in alphabet]
    return set(deletes + transposes + replaces + inserts)

def splits(word):
    word = word.lower()
    "Return a list of all possible (first, rest) pairs that comprise word."
    return [(word[:i], word[i:]) 
            for i in range(len(word)+1)]

alphabet = 'abcdefghijklmnopqrstuvwxyzäöüß'

def nocheck_find_suggested_words(word):
        
    WORD = word.lower()
    repeated_letters = []
    for i in list(''.join(s) for _, s in itertools.groupby(WORD)):
        if len(i) >= 2:
            repeated_letters.append(str(i))

    main_list = []
    combination_list = []
    for k in repeated_letters:
        for n in range(1,len(k)+1):
            combination_list.append(k[:n])
        main_list.append(combination_list)
        combination_list = []
    
    all_combination_list = list(itertools.product(*main_list))
    list_all_combination_list = list(map(list, all_combination_list))

    combination_repeated_letters_and_combination_list = []
    for i in list_all_combination_list:
        for j in [repeated_letters]:
            combination_repeated_letters_and_combination_list.append([j,i])

    def multiple_replacement(text,rep):

        try:
            rep = dict((re.escape(k), v) for k, v in rep.items())
            pattern = re.compile("|".join(rep.keys()))
            text = pattern.sub(lambda m: rep[re.escape(m.group(0))], str(text))
        except:
            pass

        return (text)

    Related_words = []
    for i in combination_repeated_letters_and_combination_list:
        Related_words.append(multiple_replacement(WORD, dict(zip(i[0],i[1]))))
        
    return(Related_words)

#################################################################################################################################################
#################################################################################################################################################
POSTCOMMENTID_CORRECTED = []
COMMENTS_CORRECTED = []

POSTCOMMENTID_INCORRECT = []
COMMENTS_INCORRECT = []
ERROR_WORDS_LIST_NEW = []
SUGGEST_WORDS_LIST_NEW = []

error_delete_element = []
suggest_delete_element = []

zipped_2 = zip(POSTCOMMENTID, COMMENTS, ERROR_WORDS_LIST, SUGGEST_WORDS_LIST)

Dict_Error_Suggestion = {}
Dic_Error_Suggestion_One_Edit = {}
Dic_Error_Suggestion_Split = {}
Dic_Error_Suggestion_Repeated_Letters = {}


for i,c,e,s in zipped_2:

    c_new = str(c)
    
    for j in range(0,len(e)):
            
        if len(make_list(s[j])) == 1:
            c_new = str(str(c_new).replace(str(e[j]), str(make_list(s[j])[0])))
            error_delete_element.append(str(e[j]))
            suggest_delete_element.append(make_list(s[j]))

        elif str(e[j]).isupper() == True:
            
            if de.check(str(e[j]).capitalize()) == True or de.check(str(e[j]).lower()) == True:
                c_new = str(str(c_new).replace(str(e[j]), str(e[j])))
                error_delete_element.append(str(e[j]))
                suggest_delete_element.append(make_list(s[j]))

        elif de.check(str(e[j]).capitalize()) == True: #and (str(make_list(s[j])[0]) == str(str(e[j]).capitalize())):
            c_new = str(str(c_new).replace(str(e[j]), str(e[j]).capitalize()))
            error_delete_element.append(str(e[j]))
            suggest_delete_element.append(make_list(s[j]))

        elif de.check(str(e[j]).upper()) == True: #and (str(make_list(s[j])[0]) == str(str(e[j]).upper())):
            c_new = str(str(c_new).replace(str(e[j]), str(e[j]).upper()))
            error_delete_element.append(str(e[j]))
            suggest_delete_element.append(make_list(s[j]))

        elif en.check(str(e[j])) == True or en.check(str(e[j]).lower()) == True:
            c_new = str(str(c_new).replace(str(e[j]), str(e[j]).capitalize()))
            error_delete_element.append(str(e[j]))
            suggest_delete_element.append(make_list(s[j]))

        else:
            
            if str(e[j]):
                
                try:
                    if not Dict_Error_Suggestion.get(str(e[j])):

                        Dict_Error_Suggestion[str(e[j])] = make_list(s[j])
                except:
                    pass

                try:
                    if find_suggested_words_repeated_letters(str(e[j])):

                        if not Dic_Error_Suggestion_Repeated_Letters.get(str(e[j])):

                            Dic_Error_Suggestion_Repeated_Letters[str(e[j])] = find_suggested_words_repeated_letters(str(e[j]))
                except:
                    pass

                try:
                    if splits_two_words(str(e[j])):

                        if not Dic_Error_Suggestion_Split.get(str(e[j])):
                    
                            Dic_Error_Suggestion_Split[str(e[j])] = splits_two_words(str(e[j]))
                except:
                    pass

                try:
                    if find_suggested_words_one_edit(str(e[j])):

                        if not Dic_Error_Suggestion_One_Edit.get(str(e[j])):

                            Dic_Error_Suggestion_One_Edit[str(e[j])] = find_suggested_words_one_edit(str(e[j]))
                except:
                    pass
            


    if len(e) == len(error_delete_element):

        POSTCOMMENTID_CORRECTED.append(str(i))
        COMMENTS_CORRECTED.append(str(c_new))
        
    else:
        
        if str(i) not in POSTCOMMENTID_INCORRECT:
            POSTCOMMENTID_INCORRECT.append(str(i))
            COMMENTS_INCORRECT.append(str(c_new))

        if error_delete_element:
            
            for e1 in error_delete_element:
                try:
                    e.remove(str(e1))
                except:
                    pass

            ERROR_WORDS_LIST_NEW.append(e)

        else:

            ERROR_WORDS_LIST_NEW.append(e)

        if suggest_delete_element:
            
            for s1 in suggest_delete_element:
                
                try:
                    
                    s.remove(str(s1))

                except:
                    
                    pass
                
            SUGGEST_WORDS_LIST_NEW.append(s)

        else:
            
            SUGGEST_WORDS_LIST_NEW.append(s)

    error_delete_element = []
    suggest_delete_element = []


zipped_3 = zip(commentID, comments)
for i,j in zipped_3:
    if i not in POSTCOMMENTID_CORRECTED and i not in POSTCOMMENTID_INCORRECT:
        POSTCOMMENTID_CORRECTED.append(str(i))
        COMMENTS_CORRECTED.append(str(j))

zipped_4 = zip(POSTCOMMENTID_INCORRECT, COMMENTS_INCORRECT, ERROR_WORDS_LIST_NEW, SUGGEST_WORDS_LIST_NEW)
for i,c,e,s in zipped_4:
    if not e:
        POSTCOMMENTID_CORRECTED.append(str(i))
        COMMENTS_CORRECTED.append(str(c))
        

post_comment_corrected = Table().with_columns('PostCommentID', POSTCOMMENTID_CORRECTED,
                                              'Comments', COMMENTS_CORRECTED)

post_comment_corrected.to_csv("CorrectedComments.csv")

POSTCOMMENTID_INCORRECT_FINAL = []
COMMENTS_INCORRECT_FINAL = []
ERROR_WORDS_LIST_FINAL = []
SUGGEST_WORDS_LIST_FINAL = []

zipped_5 = zip(POSTCOMMENTID_INCORRECT, COMMENTS_INCORRECT, ERROR_WORDS_LIST_NEW, SUGGEST_WORDS_LIST_NEW)
for i,c,e,s in zipped_5:
    if i not in POSTCOMMENTID_CORRECTED:
        POSTCOMMENTID_INCORRECT_FINAL.append(str(i))
        COMMENTS_INCORRECT_FINAL.append(str(c))
        ERROR_WORDS_LIST_FINAL.append(str(e))
        SUGGEST_WORDS_LIST_FINAL.append(str(s))

post_comment_incorrect = Table().with_columns('PostCommentID', POSTCOMMENTID_INCORRECT_FINAL,
                                              'Comments', COMMENTS_INCORRECT_FINAL,
                                              'ErrorWords', ERROR_WORDS_LIST_FINAL,
                                              'SuggestWords', SUGGEST_WORDS_LIST_FINAL)


post_comment_incorrect.to_csv("IncorrectComments.csv")

with open('C:/Users/Vahid/Desktop/New folder/Dic_Error_Suggestion_Repeated_Letters.pkl', 'wb') as f:
    pickle.dump(Dic_Error_Suggestion_Repeated_Letters, f, pickle.HIGHEST_PROTOCOL)

with open('C:/Users/Vahid/Desktop/New folder/Dic_Error_Suggestion_Split.pkl', 'wb') as f:
    pickle.dump(Dic_Error_Suggestion_Split, f, pickle.HIGHEST_PROTOCOL)

with open('C:/Users/Vahid/Desktop/New folder/Dic_Error_Suggestion_One_Edit.pkl', 'wb') as f:
    pickle.dump(Dic_Error_Suggestion_One_Edit, f, pickle.HIGHEST_PROTOCOL)

with open('C:/Users/Vahid/Desktop/New folder/Dict_Error_Suggestion.pkl', 'wb') as f:
    pickle.dump(Dict_Error_Suggestion, f, pickle.HIGHEST_PROTOCOL)


