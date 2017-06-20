import html
import re
import itertools
import pickle


with open('train_TEXT.txt', 'r', encoding='utf-8') as infile:
    TextFile = infile.readlines()

def multipleReplace(text, wordDict):
    for key in wordDict:
        text = text.replace(key, " " + str(wordDict[key])+ " ")
    return text

with open('C:/Users/Vahid/Desktop/Dict_Emojis.pkl', 'rb') as f:
    Dict_Emojis = pickle.load(f)

def cleanText(line):
   result = html.unescape(line)
   result = re.sub(r"\shttp\S+", "", result)
   result = re.sub(r"http\S+", "", result)
   result = re.sub(r"\swww\S+", "", result)
   result = re.sub(r"www\S+", "", result)
   result = re.sub(r"--\S+", "", result)
   result = re.sub(r"\s--\S+", "", result)
   result = re.sub(r"--\S+", "", result)
   result = ''.join(''.join(s)[:3] for _, s in itertools.groupby(result))
   result = result.replace("Timeline Photos", " ").replace("No Title for this Attachment", " ").replace("No Message for this Post", " ").replace("+", " ").replace("nan", " ").replace("No Comment Message", " ").replace("RP ONLINE", " ").replace("Tagesspiegel", " ")
   edited_line = multipleReplace(result.replace('\n',' ').replace("@", " ").replace('\t',' ').strip(), Dict_Emojis)
   edited_line = re.sub(r'\s+', ' ', edited_line).lower()
   return(edited_line)


#compile documents
doc_complete = []
for i in TextFile:
    doc_complete.append(cleanText(str(i)))

from nltk.corpus import stopwords 
from textblob_de import TextBlobDE as TextBlob
import string
stop = set(stopwords.words('german'))
stop.add("dass")
stop.add("mal")
exclude = set(string.punctuation) 

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(TextBlob(str(punc_free)).words.lemmatize())
    return(normalized)

doc_clean = [clean(doc).split() for doc in doc_complete]

# Importing Gensim
import gensim
from gensim import corpora

# Creating the term dictionary of our courpus, where every unique term is assigned an index.
dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel


# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=10, id2word = dictionary, passes=20)

print(ldamodel.print_topics(num_topics=10, num_words=10))


def Predict(text):

    doc_clean_words = clean(text).split()
    new_review_bow = dictionary.doc2bow(doc_clean_words)
    new_review_lda = ldamodel[new_review_bow]

    return(new_review_lda)
