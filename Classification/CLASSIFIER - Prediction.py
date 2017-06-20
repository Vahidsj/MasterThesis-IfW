import numpy as np
import html
import re
import itertools
import pickle

# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

# spacy
import spacy
nlp = spacy.load('de')


# Replacement simultaneously
def multipleReplace(text, wordDict):
    for key in wordDict:
        text = text.replace(key, " " + str(wordDict[key])+ " ")
    return text

# Loading Emojis dictionary
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
   result = result.replace("Timeline Photos", " ").replace("No Title for this Attachment", " ").replace("No Message for this Post", " ").replace("+", " ").replace("nan", " ").replace("No Comment Message", " ")
   edited_line = multipleReplace(result.replace('\n',' ').replace("@", " ").replace('\t',' ').strip(), Dict_Emojis)
   edited_line = re.sub(r'\s+', ' ', edited_line)
   
   return(edited_line)

# Loading Text Files
with open('C:/Users/Vahid/Desktop/.../PosNeg.txt', 'r', encoding='utf-8') as infile:
    Label_1 = infile.readlines()

with open('C:/Users/Vahid/Desktop/.../Neutral.txt', 'r', encoding='utf-8') as infile:
    Label_0 = infile.readlines()


# Creating DataSet
PosNeg_comments = []
Neutral_comments = []

for i in Label_1:
    comment = str(i).split("\t")[-1]
    comment_cleaned = cleanText(str(comment))
    if comment_cleaned:
        PosNeg_comments.append(cleanText(str(i)).replace("\n", ""))

for i in Label_0:
    comment = str(i).split("\t")[-1]
    comment_cleaned = cleanText(str(comment))
    if comment_cleaned:
        Neutral_comments.append(cleanText(str(i)).replace("\n", ""))


def cleanText_Tokenizer(line):
   result = html.unescape(line)
   result = re.sub(r"\shttp\S+", "", result)
   result = re.sub(r"http\S+", "", result)
   result = re.sub(r"\swww\S+", "", result)
   result = re.sub(r"www\S+", "", result)
   result = re.sub(r"--\S+", "", result)
   result = re.sub(r"\s--\S+", "", result)
   result = re.sub(r"--\S+", "", result)
   result = ''.join(''.join(s)[:3] for _, s in itertools.groupby(result))
   result = result.replace("Timeline Photos", " ").replace("No Title for this Attachment", " ").replace("No Message for this Post", " ").replace("+", " ").replace("nan", " ").replace("No Comment Message", " ")
   edited_line = multipleReplace(result.replace('\n',' ').replace("@", " ").replace('\t',' ').strip(), Dict_Emojis)
   edited_line = re.sub(r'\s+', ' ', edited_line).lower()
   word_list = []
   for token in nlp(str(edited_line)):
      word_list.append(str(token))
   return(word_list)

X_Text = []
y = []

for i in PosNeg_comments:
    X_Text.append(cleanText_Tokenizer(str(i)))
    y.append(1)

for i in Neutral_comments:
    X_Text.append(cleanText_Tokenizer(str(i)))
    y.append(0)

# Loading the Model        
model_POSNEG_NEUTRAL = Doc2Vec.load('C:/Users/Vahid/Desktop/.../TEXT/Neutral - PosNeg (1&1)/WithPost/imdb.d2v')

X = []

for i in X_Text:
    X.append(model_POSNEG_NEUTRAL.infer_vector(i))

print("DataSet is Ready!")

######################################################################################### Training Classifier using Cross Validation

# classifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score

random_state = np.random.RandomState(0)

X_array , y_label = shuffle(X, y, random_state=random_state)

SCORES = []
CLASSIFIER = []

LoR_classifier = LogisticRegression()
scores = cross_val_score(LoR_classifier, X_array, y_label, cv=10, scoring='accuracy')
print('LogisticRegression Accuracy: %.4f' %scores.mean())
SCORES.append(scores.mean())
CLASSIFIER.append('LogisticRegression')

SVC_classifier = SVC()
scores = cross_val_score(SVC_classifier, X_array, y_label, cv=10, scoring='accuracy')
print('SVC Accuracy: %.4f' %scores.mean())
SCORES.append(scores.mean())
CLASSIFIER.append('SVC')

LinearSVC_classifier = LinearSVC()
scores = cross_val_score(SVC_classifier, X_array, y_label, cv=10, scoring='accuracy')
print('LinearSVC Accuracy: %.4f' %scores.mean())
SCORES.append(scores.mean())
CLASSIFIER.append('LinearSVC')


SGD_classifier = SGDClassifier()
scores = cross_val_score(SGD_classifier, X_array, y_label, cv=10, scoring='accuracy')
print('SGDClassifier Accuracy: %.4f' %scores.mean())
SCORES.append(scores.mean())
CLASSIFIER.append('SGDClassifier')

try:
    NuSVC_classifier = NuSVC()
    scores = cross_val_score(NuSVC_classifier, X_array, y_label, cv=10, scoring='accuracy')
    print('NuSVC Accuracy: %.4f' %scores.mean())
    SCORES.append(scores.mean())
    CLASSIFIER.append('NuSVC')
except:
    pass

try:
    MultinomialNB_classifier = MultinomialNB()
    scores = cross_val_score(MultinomialNB_classifier, X_array, y_label, cv=10, scoring='accuracy')
    print('MultinomialNB Accuracy: %.4f' %scores.mean())
    SCORES.append(scores.mean())
    CLASSIFIER.append('MultinomialNB')
except:
    pass

try:
    BernoulliNB_classifier = BernoulliNB()
    scores = cross_val_score(BernoulliNB_classifier, X_array, y_label, cv=10, scoring='accuracy')
    print('BernoulliNB Accuracy: %.4f' %scores.mean())
    SCORES.append(scores.mean())
    CLASSIFIER.append('BernoulliNB')
except:
    pass

try:
    GaussianNB_classifier = GaussianNB()
    scores = cross_val_score(GaussianNB_classifier, X_array, y_label, cv=10, scoring='accuracy')
    print('GaussianNB Accuracy: %.4f' %scores.mean())
    SCORES.append(scores.mean())
    CLASSIFIER.append('GaussianNB')
except:
    pass

################################################################################# Plotting
import matplotlib.pyplot as plt
lw = 5
# plot the classifier type (x-axis) versus the cross-validated accuracy (y-axis)
x_line = np.arange(len(CLASSIFIER))
plt.bar(x_line, SCORES, color='blue', align='center', alpha=0.5)
plt.xticks(x_line, CLASSIFIER)
plt.xlabel('Classifier')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Classifier Accuracy Comparison for Neutral versus Positive-Negative')
plt.show()

################################################################################# searching for optimal tuning parameters
from sklearn.model_selection import GridSearchCV

LoR_classifier = LogisticRegression(class_weight=None, dual=False, intercept_scaling=1, random_state=random_state)

# define the parameter values that should be searched
penalty_options = ['l1', 'l2']
C_options = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
fit_intercept_options = [False, True]
tol_options = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e+1, 1e+2]

# create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(penalty=penalty_options, C=C_options, fit_intercept=fit_intercept_options, tol=tol_options)
print(param_grid)


# instantiate and fit the grid
grid = GridSearchCV(LoR_classifier, param_grid, cv=10, scoring='accuracy')
grid.fit(X_array, y_label)

# examine the best model
print(grid.best_score_)
print(grid.best_params_)

################################################################################# prediction
from datascience import *
import os
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_array, y_label, test_size=0.2, random_state=random_state)

LoR_classifier = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, penalty='l2', random_state=random_state, tol=0.0001)
LoR_classifier.fit(X_train, y_train)

for filename in os.listdir('C:/Users/Vahid/Desktop/...'):

    COMMENTSTABLE = Table.read_table("C:/Users/Vahid/Desktop/.../" + str(filename), encoding="ISO-8859-1")
    ID = COMMENTSTABLE.column(0)
    PostCreatedTime = COMMENTSTABLE.column(1)
    PostKeyword = COMMENTSTABLE.column(2)
    PostMessage = COMMENTSTABLE.column(3)
    PostAttachment = COMMENTSTABLE.column(4)
    CommentCreatedTime = COMMENTSTABLE.column(5)
    CommentUserName = COMMENTSTABLE.column(6)
    CommentUserID = COMMENTSTABLE.column(7)
    CommentMessage = COMMENTSTABLE.column(8)

    Id = []
    P_C_T = []
    P_K = []
    P_M = []
    P_A = []
    C_C_T = []
    C_U_N = []
    C_U_I = []
    C_M = []
    
    zipped= zip(ID, PostCreatedTime, PostKeyword, PostMessage, PostAttachment, CommentCreatedTime, CommentUserName, CommentUserID, CommentMessage)
    comment_array = []
    for iD, p_c_t, p_k, p_m, p_a, c_c_t, c_u_n, c_u_i, c_m in zipped:
        
        if cleanText(c_m):
            
            Id.append(str(iD))
            P_C_T.append(str(p_c_t))
            P_K.append(str(p_k))
            P_M.append(str(p_m))
            P_A.append(str(p_a))
            C_C_T.append(str(c_c_t))
            C_U_N.append(str(c_u_n))
            C_U_I.append(str(c_u_i))
            C_M.append(cleanText(str(c_m)))
            
            Comment_Tokenized = cleanText_Tokenizer(str(p_m) + str(p_a) + str(c_m))
            comment_array.append(model_POSNEG_NEUTRAL.infer_vector(Comment_Tokenized))

    print('Comments are ready to be predicted!')

    predicted_label = LoR_classifier.predict(comment_array)

    PN_Id = []
    PN_P_C_T = []
    PN_P_K = []
    PN_P_M = []
    PN_P_A = []
    PN_C_C_T = []
    PN_C_U_N = []
    PN_C_U_I = []
    PN_C_M = []
    PN_predicted_label = []

    N_Id = []
    N_P_C_T = []
    N_P_K = []
    N_P_M = []
    N_P_A = []
    N_C_C_T = []
    N_C_U_N = []
    N_C_U_I = []
    N_C_M = []
    N_predicted_label = []
    
    zipped= zip(Id, P_C_T, P_K, P_M, P_A, C_C_T, C_U_N, C_U_I, C_M, predicted_label)
    for iD, p_c_t, p_k, p_m, p_a, c_c_t, c_u_n, c_u_i, c_m, p_l in zipped:
        
        if p_l == 1:
            
            PN_Id.append(str(iD))
            PN_P_C_T.append(str(p_c_t))
            PN_P_K.append(str(p_k))
            PN_P_M.append(str(p_m))
            PN_P_A.append(str(p_a))
            PN_C_C_T.append(str(c_c_t))
            PN_C_U_N.append(str(c_u_n))
            PN_C_U_I.append(str(c_u_i))
            PN_C_M.append(cleanText(str(c_m)))
            PN_predicted_label.append(p_l)

        elif p_l == 0:
            
            N_Id.append(str(iD))
            N_P_C_T.append(str(p_c_t))
            N_P_K.append(str(p_k))
            N_P_M.append(str(p_m))
            N_P_A.append(str(p_a))
            N_C_C_T.append(str(c_c_t))
            N_C_U_N.append(str(c_u_n))
            N_C_U_I.append(str(c_u_i))
            N_C_M.append(cleanText(str(c_m)))
            N_predicted_label.append(p_l)
    

    TABLE_POSNEG = Table().with_columns('PostCommentID', PN_Id,
                                        'Post Created-Time', PN_P_C_T,
                                        'Post Keyword', PN_P_K,
                                        'Post Message', PN_P_M,
                                        'Post Attachment', PN_P_A,
                                        'Comment Created-Time', PN_C_C_T,
                                        'Comment UserName', PN_C_U_N,
                                        'Comment UserID', PN_C_U_I,
                                        'Comment Message', PN_C_M)

    TABLE_NEUTRAL = Table().with_columns('PostCommentID', N_Id,
                                        'Post Created-Time', N_P_C_T,
                                        'Post Keyword', N_P_K,
                                        'Post Message', N_P_M,
                                        'Post Attachment', N_P_A,
                                        'Comment Created-Time', N_C_C_T,
                                        'Comment UserName', N_C_U_N,
                                        'Comment UserID', N_C_U_I,
                                        'Comment Message', N_C_M,
                                        'Predicted Label', N_predicted_label)

    TABLE_POSNEG.to_csv('C:/Users/Vahid/Desktop/.../'+(str(filename).split("-")[0])+"POSNEG.csv")
    TABLE_NEUTRAL.to_csv('C:/Users/Vahid/Desktop/.../'+(str(filename).split("-")[0])+"NEUTRAL.csv")
    print(str(filename))
            
# Loading Text Files
with open('C:/Users/Vahid/Desktop/..../Positive.txt', 'r', encoding='utf-8') as infile:
    Label_1 = infile.readlines()

with open('C:/Users/Vahid/Desktop/.../Negative.txt', 'r', encoding='utf-8') as infile:
    Label__1 = infile.readlines()


# Creating DataSet
Pos_comments = []
Neg_comments = []

for i in Label_1:
    comment = str(i).split("\t")[-1]
    comment_cleaned = cleanText(str(comment))
    if comment_cleaned:
        Pos_comments.append(cleanText(str(i)).replace("\n", ""))

for i in Label__1:
    comment = str(i).split("\t")[-1]
    comment_cleaned = cleanText(str(comment))
    if comment_cleaned:
        Neg_comments.append(cleanText(str(i)).replace("\n", ""))

X_Text = []
y = []

for i in Pos_comments:
    X_Text.append(cleanText_Tokenizer(str(i)))
    y.append(1)

for i in Neg_comments:
    X_Text.append(cleanText_Tokenizer(str(i)))
    y.append(-1)

# Loading the Model        
model_POS_NEG = Doc2Vec.load('C:/Users/Vahid/Desktop/.../Positive - Negative (1&1)/imdb.d2v')

X = []

for i in X_Text:
    X.append(model_POS_NEG.infer_vector(i))

print("DataSet is Ready!")

######################################################################################### Training Classifier using Cross Validation

X_array , y_label = shuffle(X, y, random_state=random_state)

SCORES = []
CLASSIFIER = []

LoR_classifier = LogisticRegression()
scores = cross_val_score(LoR_classifier, X_array, y_label, cv=10, scoring='accuracy')
print('LogisticRegression Accuracy: %.4f' %scores.mean())
SCORES.append(scores.mean())
CLASSIFIER.append('LogisticRegression')

SVC_classifier = SVC()
scores = cross_val_score(SVC_classifier, X_array, y_label, cv=10, scoring='accuracy')
print('SVC Accuracy: %.4f' %scores.mean())
SCORES.append(scores.mean())
CLASSIFIER.append('SVC')

LinearSVC_classifier = LinearSVC()
scores = cross_val_score(SVC_classifier, X_array, y_label, cv=10, scoring='accuracy')
print('LinearSVC Accuracy: %.4f' %scores.mean())
SCORES.append(scores.mean())
CLASSIFIER.append('LinearSVC')


SGD_classifier = SGDClassifier()
scores = cross_val_score(SGD_classifier, X_array, y_label, cv=10, scoring='accuracy')
print('SGDClassifier Accuracy: %.4f' %scores.mean())
SCORES.append(scores.mean())
CLASSIFIER.append('SGDClassifier')

try:
    NuSVC_classifier = NuSVC()
    scores = cross_val_score(NuSVC_classifier, X_array, y_label, cv=10, scoring='accuracy')
    print('NuSVC Accuracy: %.4f' %scores.mean())
    SCORES.append(scores.mean())
    CLASSIFIER.append('NuSVC')
except:
    pass

try:
    MultinomialNB_classifier = MultinomialNB()
    scores = cross_val_score(MultinomialNB_classifier, X_array, y_label, cv=10, scoring='accuracy')
    print('MultinomialNB Accuracy: %.4f' %scores.mean())
    SCORES.append(scores.mean())
    CLASSIFIER.append('MultinomialNB')
except:
    pass

try:
    BernoulliNB_classifier = BernoulliNB()
    scores = cross_val_score(BernoulliNB_classifier, X_array, y_label, cv=10, scoring='accuracy')
    print('BernoulliNB Accuracy: %.4f' %scores.mean())
    SCORES.append(scores.mean())
    CLASSIFIER.append('BernoulliNB')
except:
    pass

try:
    GaussianNB_classifier = GaussianNB()
    scores = cross_val_score(GaussianNB_classifier, X_array, y_label, cv=10, scoring='accuracy')
    print('GaussianNB Accuracy: %.4f' %scores.mean())
    SCORES.append(scores.mean())
    CLASSIFIER.append('GaussianNB')
except:
    pass

################################################################################# Plotting

lw = 5
# plot the classifier type (x-axis) versus the cross-validated accuracy (y-axis)
x_line = np.arange(len(CLASSIFIER))
plt.bar(x_line, SCORES, color='blue', align='center', alpha=0.5)
plt.xticks(x_line, CLASSIFIER)
plt.xlabel('Classifier')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Classifier Accuracy Comparison for Positive versus Negative')
plt.show()

################################################################################# searching for optimal tuning parameters

LoR_classifier = LogisticRegression(class_weight=None, dual=False, intercept_scaling=1, random_state=random_state)

# define the parameter values that should be searched
penalty_options = ['l1', 'l2']
C_options = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
fit_intercept_options = [False, True]
tol_options = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e+1, 1e+2]

# create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(penalty=penalty_options, C=C_options, fit_intercept=fit_intercept_options, tol=tol_options)
print(param_grid)


# instantiate and fit the grid
grid = GridSearchCV(LoR_classifier, param_grid, cv=10, scoring='accuracy')
grid.fit(X_array, y_label)

# examine the best model
print(grid.best_score_)
print(grid.best_params_)

################################################################################# prediction

X_train, X_test, y_train, y_test = train_test_split(X_array, y_label, test_size=0.2, random_state=random_state)

LoR_classifier = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, penalty='l1', random_state=random_state, tol=0.1)
LoR_classifier.fit(X_train, y_train)
print(LoR_classifier.score(X_test, y_test))

files_zipped = zip(os.listdir('C:/Users/Vahid/Desktop/.../POSNEG')[10:], os.listdir('C:/Users/Vahid/Desktop/.../Neutral')[10:])
for filename_PN, filename_N in files_zipped:

    COMMENTSTABLE = Table.read_table('C:/Users/Vahid/Desktop/.../'+ str(filename_PN), encoding="ISO-8859-1")
    COMMENTSTABLE_N = Table.read_table('C:/Users/Vahid/Desktop/.../'+ str(filename_N), encoding="ISO-8859-1")
    
    ID = COMMENTSTABLE.column(0)
    PostCreatedTime = COMMENTSTABLE.column(1)
    PostKeyword = COMMENTSTABLE.column(2)
    PostMessage = COMMENTSTABLE.column(3)
    PostAttachment = COMMENTSTABLE.column(4)
    CommentCreatedTime = COMMENTSTABLE.column(5)
    CommentUserName = COMMENTSTABLE.column(6)
    CommentUserID = COMMENTSTABLE.column(7)
    CommentMessage = COMMENTSTABLE.column(8)

    
    N_Id = COMMENTSTABLE_N.column(0)
    N_P_C_T = COMMENTSTABLE_N.column(1)
    N_P_K = COMMENTSTABLE_N.column(2)
    N_P_M = COMMENTSTABLE_N.column(3)
    N_P_A = COMMENTSTABLE_N.column(4)
    N_C_C_T = COMMENTSTABLE_N.column(5)
    N_C_U_N = COMMENTSTABLE_N.column(6)
    N_C_U_I = COMMENTSTABLE_N.column(7)
    N_C_M = COMMENTSTABLE_N.column(8)
    N_predicted_label = COMMENTSTABLE_N.column(9)

    Id = []
    P_C_T = []
    P_K = []
    P_M = []
    P_A = []
    C_C_T = []
    C_U_N = []
    C_U_I = []
    C_M = []
    
    zipped= zip(ID, PostCreatedTime, PostKeyword, PostMessage, PostAttachment, CommentCreatedTime, CommentUserName, CommentUserID, CommentMessage)
    comment_array = []
    for iD, p_c_t, p_k, p_m, p_a, c_c_t, c_u_n, c_u_i, c_m in zipped:
        
        if cleanText(c_m):
            
            Id.append(str(iD))
            P_C_T.append(str(p_c_t))
            P_K.append(str(p_k))
            P_M.append(str(p_m))
            P_A.append(str(p_a))
            C_C_T.append(str(c_c_t))
            C_U_N.append(str(c_u_n))
            C_U_I.append(str(c_u_i))
            C_M.append(cleanText(str(c_m)))
            
            Comment_Tokenized = cleanText_Tokenizer(str(p_m) + str(p_a) + str(c_m))
            comment_array.append(model_POS_NEG.infer_vector(Comment_Tokenized))

    print('Comments are ready to be predicted!')

    predicted_label = LoR_classifier.predict(comment_array)
    
    A_Id = Id + list(N_Id)
    A_P_C_T = P_C_T + list(N_P_C_T)
    A_P_K = P_K + list(N_P_K)
    A_P_M = P_M + list(N_P_M)
    A_P_A = P_A + list(N_P_A)
    A_C_C_T = C_C_T + list(N_C_C_T)
    A_C_U_N = C_U_N + list(N_C_U_N)
    A_C_U_I = C_U_I + list(N_C_U_I)
    A_C_M = C_M + list(N_C_M)
    A_predicted_label = list(predicted_label) + list(N_predicted_label)
    
    TABLE = Table().with_columns('PostCommentID', A_Id,
                                 'Post Created-Time', A_P_C_T,
                                 'Post Keyword', A_P_K,
                                 'Post Message', A_P_M,
                                 'Post Attachment', A_P_A,
                                 'Comment Created-Time', A_C_C_T,
                                 'Comment UserName', A_C_U_N,
                                 'Comment UserID', A_C_U_I,
                                 'Comment Message', A_C_M,
                                 'Predicted Label', A_predicted_label)
    
    TABLE.to_csv('C:/Users/Vahid/Desktop/.../'+str(filename_N)[:-11]+".csv")
    print(str(filename_N)[:-11])
