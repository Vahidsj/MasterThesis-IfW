import numpy as np
import html
import re
import itertools
import pickle

# Replacement simultaneously
def multipleReplace(text, wordDict):
    for key in wordDict:
        text = text.replace(key, " " + str(wordDict[key])+ " ")
    return text

# Loading Emojis dictionary
with open('C:/Users/.../Dict_Emojis.pkl', 'rb') as f:
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
with open('Positive.txt', 'r', encoding='utf-8') as infile:
    Label_1 = infile.readlines()

with open('Negative.txt', 'r', encoding='utf-8') as infile:
    Label_0 = infile.readlines()


# Splitting Data to Training and Test Set
Pos_comments = []
Neg_comments = []

for i in Label_1:
    comment = str(i).split("\t")[-1]
    comment_cleaned = cleanText(str(comment))
    if comment_cleaned:
        Pos_comments.append(cleanText(str(i)).replace("\n", ""))
    else:
        print(comment)

for i in Label_0:
    comment = str(i).split("\t")[-1]
    comment_cleaned = cleanText(str(comment))
    if comment_cleaned:
        Neg_comments.append(cleanText(str(i)).replace("\n", ""))
    else:
        print(comment)

for i in range(5):
    np.random.shuffle(Pos_comments)

train_Pos_comments, test_Pos_comments = Pos_comments[:round(len(Pos_comments)*(80/100))], Pos_comments[round(len(Pos_comments)*(80/100)):]

for i in range(5):
    np.random.shuffle(Neg_comments)

train_Neg_comments, test_Neg_comments = Neg_comments[:round(len(Neg_comments)*(80/100))], Neg_comments[round(len(Neg_comments)*(80/100)):]


# Save Data as TEXT File
with open('train_1.txt', 'w', encoding='utf-8') as f:
    for i in train_Pos_comments:
        f.write(str(i)+"\n")
        

with open('test_1.txt', 'w', encoding='utf-8') as f:
    for i in test_Pos_comments:
        f.write(str(i)+"\n")
        

with open('train_0.txt', 'w', encoding='utf-8') as f:
    for i in train_Neg_comments:
        f.write(str(i)+"\n")
        

with open('test_0.txt', 'w', encoding='utf-8') as f:
    for i in test_Neg_comments:
        f.write(str(i)+"\n")

######################################################################################### Training Doc2Vec Model

        
# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

# spacy
import spacy
nlp = spacy.load('de')

# pickle
import pickle

# random shuffle
from random import shuffle

# numpy
import numpy

import logging
import sys

with open('train_1.txt', 'r', encoding='utf-8') as infile:
    train_1 = infile.readlines()

with open('test_1.txt', 'r', encoding='utf-8') as infile:
    test_1 = infile.readlines()

with open('train_0.txt', 'r', encoding='utf-8') as infile:
    train_0 = infile.readlines()

with open('test_0.txt', 'r', encoding='utf-8') as infile:
    test_0 = infile.readlines()


len_train_1 = len(train_1)
len_train_0 = len(train_0)
len_test_1 = len(test_1)
len_test_0 = len(test_0)

log = logging.getLogger()
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)


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
   edited_line = re.sub(r'\s+', ' ', edited_line).lower()
   word_list = []
   for token in nlp(str(edited_line)):
      word_list.append(str(token))
   return(word_list)

class TaggedLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield TaggedDocument(cleanText(utils.to_unicode(line)), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(TaggedDocument(cleanText(utils.to_unicode(line)), [prefix + '_%s' % item_no]))
        return(self.sentences)

    def sentences_perm(self):
        shuffle(self.sentences)
        return(self.sentences)
        

log.info('source load')
sources = {'train_1.txt':'TRAIN_POS', 'test_1.txt':'TEST_POS', 'train_0.txt':'TRAIN_NEG', 'test_0.txt':'TEST_NEG', 'train-unsup.txt':'TRAIN_UNS'}

log.info('TaggedDocument')
sentences = TaggedLineSentence(sources)

log.info('D2V')
model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=7, workers=5)
model.build_vocab(sentences.to_array())

log.info('Epoch')
for epoch in range(10):
	log.info('EPOCH: {}'.format(epoch))
	model.train(sentences.sentences_perm())

log.info('Model Save')
model.save('./imdb.d2v')
model = Doc2Vec.load('./imdb.d2v')

log.info('Sentiment')
train_arrays = numpy.zeros((len_train_1 + len_train_0, 100))
train_labels = numpy.zeros(len_train_1 + len_train_0)

for i in range(len_train_1):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_pos]
    train_labels[i] = 1

for i in range(len_train_0):
    prefix_train_neutral = 'TRAIN_NEG_' + str(i)
    train_arrays[len_train_1 + i] = model.docvecs[prefix_train_neutral]
    train_labels[len_train_1 + i] = 0

print(train_labels)

test_arrays = numpy.zeros((len_test_1 + len_test_0, 100))
test_labels = numpy.zeros(len_test_1 + len_test_0)

for i in range(len_test_1):
    prefix_test_pos = 'TEST_POS_' + str(i)
    test_arrays[i] = model.docvecs[prefix_test_pos]
    test_labels[i] = 1

for i in range(len_test_0):
    prefix_test_neutral = 'TEST_NEG_' + str(i)
    test_arrays[len_test_1 + i] = model.docvecs[prefix_test_neutral]
    test_labels[len_test_1 + i] = 0

log.info('Fitting')

# classifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.utils import shuffle
import numpy as np

random_state = np.random.RandomState(0)
train_arrays, train_labels = shuffle(train_arrays, train_labels)
test_arrays, test_labels = shuffle(test_arrays, test_labels)

LoR_classifier = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, penalty='l2', multi_class='ovr', random_state=random_state, tol=0.0001)
LoR_classifier.fit(train_arrays, train_labels)
print('LogisticRegression Accuracy: %.2f' %LoR_classifier.score(test_arrays, test_labels))

SVC_classifier = SVC(probability=True)
SVC_classifier.fit(train_arrays, train_labels)
print('SVC Accuracy: %.2f' %SVC_classifier.score(test_arrays, test_labels))

LinearSVC_classifier = LinearSVC(multi_class='ovr')
LinearSVC_classifier.fit(train_arrays, train_labels)
print('LinearSVC Accuracy: %.2f' %LinearSVC_classifier.score(test_arrays, test_labels))

SGD_classifier = SGDClassifier()
SGD_classifier.fit(train_arrays, train_labels)
print('SGDClassifier Accuracy: %.2f' %SGD_classifier.score(test_arrays, test_labels))


try:
    NuSVC_classifier = NuSVC()
    NuSVC_classifier.fit(train_arrays, train_labels)
    print('NuSVC Accuracy: %.2f' %NuSVC_classifier.score(test_arrays, test_labels))
except:
    pass

try:
    MultinomialNB_classifier = MultinomialNB()
    MultinomialNB_classifier.fit(train_arrays, train_labels)
    print('MultinomialNB Accuracy: %.2f' %MultinomialNB_classifier.score(test_arrays, test_labels))
except:
    pass

try:
    BernoulliNB_classifier = BernoulliNB()
    BernoulliNB_classifier.fit(train_arrays, train_labels)
    print('BernoulliNB Accuracy: %.2f' %BernoulliNB_classifier.score(test_arrays, test_labels))
except:
    pass

try:
    GaussianNB_classifier = GaussianNB()
    GaussianNB_classifier.fit(train_arrays, train_labels)
    print('GaussianNB Accuracy: %.2f' %GaussianNB_classifier.score(test_arrays, test_labels))
except:
    pass

################################################################################# Confusion_matrix

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


#############################################################################  Confusion_matrix(y_true, y_pred, labels=None, sample_weight=None) for LogisticRegression

y_test = test_labels
y_pred = LoR_classifier.predict(test_arrays)
class_names = np.unique(test_labels)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization for LogisticRegression')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix for LogisticRegression')

plt.show()

#############################################################################  Confusion_matrix(y_true, y_pred, labels=None, sample_weight=None) SVC

y_test = test_labels
y_pred = SVC_classifier.predict(test_arrays)
class_names = np.unique(test_labels)
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization for SVC')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix for SVC')

plt.show()
#############################################################################  Confusion_matrix(y_true, y_pred, labels=None, sample_weight=None) LinearSVC

y_test = test_labels
y_pred = LinearSVC_classifier.predict(test_arrays)
class_names = np.unique(test_labels)
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization for LinearSVC')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix for LinearSVC')

plt.show()
########################################################################## ROC and Precision-Recall Curve for LogisticRegression

from sklearn.metrics import precision_recall_curve

lw=5
pred_probas = LoR_classifier.predict_proba(test_arrays)[:,1]

fpr,tpr,_ = roc_curve(test_labels, pred_probas)
roc_auc = auc(fpr,tpr)
print("Area Under Curve: %0.2f" % roc_auc)
plt.plot(fpr,tpr, color='cornflowerblue' , lw=lw, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('ROC for LogisticRegression: AUC=%0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.show()

pred_probas = LoR_classifier.predict_proba(test_arrays)[:,1]
precision, recall, thresholds = precision_recall_curve(test_labels, pred_probas)
area = auc(recall, precision)
print("Area Under Curve: %0.2f" % area)
plt.clf()
plt.plot(recall, precision, color='turquoise', lw=lw, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall for LogisticRegression: AUC=%0.2f' % area)
plt.legend(loc="lower left")
plt.show()

########################################################################## ROC and Precision-Recall Curve for SVC

pred_probas = SVC_classifier.predict_proba(test_arrays)[:,1]

fpr,tpr,_ = roc_curve(test_labels, pred_probas)
roc_auc = auc(fpr,tpr)
print("Area Under Curve: %0.2f" % roc_auc)
plt.plot(fpr,tpr, color='cornflowerblue', lw=lw, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('ROC for SVC: AUC=%0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.show()

pred_probas = SVC_classifier.predict_proba(test_arrays)[:,1]
precision, recall, thresholds = precision_recall_curve(test_labels, pred_probas)
area = auc(recall, precision)
print("Area Under Curve: %0.2f" % area)
plt.clf()
plt.plot(recall, precision, color='turquoise', lw=lw, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall for SVC: AUC=%0.2f' % area)
plt.legend(loc="lower left")
plt.show()
