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


# Text Cleaning
def cleanText_Split(line):
    
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
with open('Pos.txt', 'r', encoding='utf-8') as infile:
    Label_1 = infile.readlines()

with open('Neg.txt', 'r', encoding='utf-8') as infile:
    Label__1 = infile.readlines()

with open('Neutral.txt', 'r', encoding='utf-8') as infile:
    Label_0 = infile.readlines()


# Splitting Data to Training and Test Set
Pos_comments = []
Neg_comments = []
Neutral_comments = []

for i in Label_1:
    
    comment = str(i).split("\t")[-1]
    comment_cleaned = cleanText_Split(str(comment))
    if comment_cleaned:
        Pos_comments.append(cleanText_Split(str(i)).replace("\n", ""))
    else:
        print(comment)

for i in Label__1:
    
    comment = str(i).split("\t")[-1]
    comment_cleaned = cleanText_Split(str(comment))
    if comment_cleaned:
        Neg_comments.append(cleanText_Split(str(i)).replace("\n", ""))
    else:
        print(comment)

for i in Label_0:
    
    comment = str(i).split("\t")[-1]
    comment_cleaned = cleanText_Split(str(comment))
    if comment_cleaned:
        Neutral_comments.append(cleanText_Split(str(i)).replace("\n", ""))
    else:
        print(comment)

for i in range(5):
    
    np.random.shuffle(Pos_comments)

train_Pos_comments, test_Pos_comments = Pos_comments[:round(len(Pos_comments)*(80/100))], Pos_comments[round(len(Pos_comments)*(80/100)):]

for i in range(5):
    
    np.random.shuffle(Neg_comments)

train_Neg_comments, test_Neg_comments = Neg_comments[:round(len(Neg_comments)*(80/100))], Neg_comments[round(len(Neg_comments)*(80/100)):]

for i in range(5):
    
    np.random.shuffle(Neutral_comments)

train_Neutral_comments, test_Neutral_comments = Neutral_comments[:round(len(Neutral_comments)*(80/100))], Neutral_comments[round(len(Neutral_comments)*(80/100)):]

# Save Data as TEXT File
with open('train_1.txt', 'w', encoding='utf-8') as f:
    for i in train_Pos_comments:
        f.write(str(i)+"\n")

with open('test_1.txt', 'w', encoding='utf-8') as f:
    for i in test_Pos_comments:
        f.write(str(i)+"\n")

with open('train__1.txt', 'w', encoding='utf-8') as f:
    for i in train_Neg_comments:
        f.write(str(i)+"\n")

with open('test__1.txt', 'w', encoding='utf-8') as f:
    for i in test_Neg_comments:
        f.write(str(i)+"\n")

with open('train_0.txt', 'w', encoding='utf-8') as f:
    for i in train_Neutral_comments:
        f.write(str(i)+"\n")

with open('test_0.txt', 'w', encoding='utf-8') as f:
    for i in test_Neutral_comments:
        f.write(str(i)+"\n")


######################################################################################################### Training Doc2Vec Model

# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

# spacy
import spacy
nlp = spacy.load('de')

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

with open('train__1.txt', 'r', encoding='utf-8') as infile:
    train__1 = infile.readlines()

with open('test__1.txt', 'r', encoding='utf-8') as infile:
    test__1 = infile.readlines()


len_train_1 = len(train_1)
len_train_0 = len(train_0)
len_train__1 = len(train__1)
len_test_1 = len(test_1)
len_test_0 = len(test_0)
len_test__1 = len(test__1)

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
sources = {'train_1.txt':'TRAIN_POS', 'test_1.txt':'TEST_POS', 'train_0.txt':'TRAIN_NEUTRAL', 'test_0.txt':'TEST_NEUTRAL', 'train__1.txt':'TRAIN_NEG', 'test__1.txt':'TEST_NEG', 'train-unsup.txt':'TRAIN_UNS'}

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
train_arrays = numpy.zeros((len_train_1 + len_train_0 + len_train__1, 100))
train_labels = numpy.zeros(len_train_1 + len_train_0 + len_train__1)

for i in range(len_train_1):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_pos]
    train_labels[i] = 1

for i in range(len_train_0):
    prefix_train_neutral = 'TRAIN_NEUTRAL_' + str(i)
    train_arrays[len_train_1 + i] = model.docvecs[prefix_train_neutral]
    train_labels[len_train_1 + i] = 0

for i in range(len_train__1):
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    train_arrays[len_train_1 + len_train_0 + i] = model.docvecs[prefix_train_neg]
    train_labels[len_train_1 + len_train_0 + i] = -1

print(train_labels)

test_arrays = numpy.zeros((len_test_1 + len_test_0 + len_test__1, 100))
test_labels = numpy.zeros(len_test_1 + len_test_0 + len_test__1)

for i in range(len_test_1):
    prefix_test_pos = 'TEST_POS_' + str(i)
    test_arrays[i] = model.docvecs[prefix_test_pos]
    test_labels[i] = 1

for i in range(len_test_0):
    prefix_test_neutral = 'TEST_NEUTRAL_' + str(i)
    test_arrays[len_test_1 + i] = model.docvecs[prefix_test_neutral]
    test_labels[len_test_1 + i] = 0

for i in range(len_test__1):
    prefix_test_neg = 'TEST_NEG_' + str(i)
    test_arrays[len_test_1 + len_test_0 + i] = model.docvecs[prefix_test_neg]
    test_labels[len_test_1 + len_test_0 + i] = -1

log.info('Fitting')

# classifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.utils import shuffle
import numpy as np

random_state = np.random.RandomState(0)
train_arrays, train_labels = shuffle(train_arrays, train_labels)

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

#############################################################################  ROC Curve for LogisticRegression

# Binarize the output
y_test = label_binarize(test_labels, classes=np.unique(test_labels))
n_classes = y_test.shape[1]

# Learn to predict each class against the other
classifier = OneVsRestClassifier(LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, penalty='l2',
                                                    random_state=random_state, tol=0.0001))

y_score = classifier.fit(train_arrays, train_labels).decision_function(test_arrays)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# Plot ROC curves for the multiclass problem

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
lw = 2
for i, color, j in zip(range(n_classes), colors, np.unique(test_labels)):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(j, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class for LogisticRegression')
plt.legend(loc="lower right")
plt.show()

############################################################################# ROC Curve for SVC

#Binarize the output
y_test = label_binarize(test_labels, classes=np.unique(test_labels))
n_classes = y_test.shape[1]

#Learn to predict each class against the other
classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True,
                                 random_state=random_state))

y_score = classifier.fit(train_arrays, train_labels).decision_function(test_arrays)

#Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

#Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


#Plot ROC curves for the multiclass problem

#Compute macro-average ROC curve and ROC area

#First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

#Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

#Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

#Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
lw = 2
for i, color, j in zip(range(n_classes), colors, np.unique(test_labels)):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(j, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class for SVC')
plt.legend(loc="lower right")
plt.show()

############################################################################# ROC Curve for LinearSVC

# Binarize the output
y_test = label_binarize(test_labels, classes=np.unique(test_labels))
n_classes = y_test.shape[1]

# Learn to predict each class against the other
classifier = OneVsRestClassifier(LinearSVC(multi_class='ovr', random_state=random_state))
y_score = classifier.fit(train_arrays, train_labels).decision_function(test_arrays)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# Plot ROC curves for the multiclass problem

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
lw = 2
for i, color, j in zip(range(n_classes), colors, np.unique(test_labels)):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(j, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class for LinearSVC')
plt.legend(loc="lower right")
plt.show()

############################################################################# Precision-Recall Curve for Logistic Regression
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
lw = 2

# Binarize the output
y_test = label_binarize(test_labels, classes=np.unique(test_labels))
n_classes = y_test.shape[1]

# Run classifier
classifier = OneVsRestClassifier(LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, penalty='l2', multi_class='ovr',
                                                    random_state=random_state, tol=0.0001))

y_score = classifier.fit(train_arrays, train_labels).decision_function(test_arrays)

# Compute Precision-Recall and plot curve
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

# Compute micro-average ROC curve and ROC area
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
    y_score.ravel())
average_precision["micro"] = average_precision_score(y_test, y_score,
                                                     average="micro")



# Plot Precision-Recall curve for each class
plt.clf()
plt.plot(recall["micro"], precision["micro"], color='gold', lw=lw,
         label='micro-average Precision-recall curve (area = {0:0.2f})'
               ''.format(average_precision["micro"]))
for i, color, j in zip(range(n_classes), colors, np.unique(test_labels)):
    plt.plot(recall[i], precision[i], color=color, lw=lw,
             label='Precision-recall curve of class {0} (area = {1:0.2f})'
                   ''.format(j, average_precision[i]))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class for LogisticRegression')
plt.legend(loc="lower right")
plt.show()


############################################################################# Precision-Recall Curve for SVC

# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
lw = 2

# Binarize the output
y_test = label_binarize(test_labels, classes=np.unique(test_labels))
n_classes = y_test.shape[1]

# Run classifier
classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True,
                                 random_state=random_state))

y_score = classifier.fit(train_arrays, train_labels).decision_function(test_arrays)

# Compute Precision-Recall and plot curve
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

# Compute micro-average ROC curve and ROC area
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
    y_score.ravel())
average_precision["micro"] = average_precision_score(y_test, y_score,
                                                     average="micro")

# Plot Precision-Recall curve for each class
plt.clf()
plt.plot(recall["micro"], precision["micro"], color='gold', lw=lw,
         label='micro-average Precision-recall curve (area = {0:0.2f})'
               ''.format(average_precision["micro"]))
for i, color, j in zip(range(n_classes), colors, np.unique(test_labels)):
    plt.plot(recall[i], precision[i], color=color, lw=lw,
             label='Precision-recall curve of class {0} (area = {1:0.2f})'
                   ''.format(j, average_precision[i]))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class for SVC')
plt.legend(loc="lower right")
plt.show()


############################################################################# Precision-Recall Curve for LinearSVC

# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
lw = 2

# Binarize the output
y_test = label_binarize(test_labels, classes=np.unique(test_labels))
n_classes = y_test.shape[1]

# Run classifier
classifier = OneVsRestClassifier(LinearSVC(random_state=random_state, multi_class='ovr'))

y_score = classifier.fit(train_arrays, train_labels).decision_function(test_arrays)

# Compute Precision-Recall and plot curve
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

# Compute micro-average ROC curve and ROC area
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
    y_score.ravel())
average_precision["micro"] = average_precision_score(y_test, y_score,
                                                     average="micro")


# Plot Precision-Recall curve for each class
plt.clf()
plt.plot(recall["micro"], precision["micro"], color='gold', lw=lw,
         label='micro-average Precision-recall curve (area = {0:0.2f})'
               ''.format(average_precision["micro"]))
for i, color, j in zip(range(n_classes), colors, np.unique(test_labels)):
    plt.plot(recall[i], precision[i], color=color, lw=lw,
             label='Precision-recall curve of class {0} (area = {1:0.2f})'
                   ''.format(j, average_precision[i]))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class for LinearSVC')
plt.legend(loc="lower right")
plt.show()
