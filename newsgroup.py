import sys
import itertools
import matplotlib.pyplot as plt
import numpy as np
import nltk

from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
from sklearn.decomposition import NMF
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier

categories = ['comp.graphics',
              'comp.os.ms-windows.misc',
              'comp.sys.ibm.pc.hardware',
              'comp.sys.mac.hardware',
              'rec.autos',
              'rec.motorcycles',
              'rec.sport.baseball',
              'rec.sport.hockey']

mult_classes = ['comp.sys.ibm.pc.hardware',
                'comp.sys.mac.hardware',
                'misc.forsale',
                'soc.religion.christian']

bin_classes = ['Computer Technology', 'Recreational Activity']

def plot_histogram(newsgroups_train):
    '''
    Plot the document size per class histogram
    '''
    class_size = np.bincount(newsgroups_train.target)
    # print(class_size)
    # [584, 591, 590, 578, 594, 598, 597, 600]

    n_groups = 8
    bar_width = 0.35
    opacity = 0.4
    plt.bar(range(n_groups), class_size, width=bar_width, alpha=opacity, color='b')

    plt.xlabel('Class')
    plt.ylabel('Number of Documnets')
    plt.title('Number of training documents per class')

    plt.show()

class StemTokenizer(object):
    '''
    override the tokenizer in CountVectorizer
    with stemming and reomoving punctuation
    '''
    def __init__(self):
        self.tokenizer = CountVectorizer().build_tokenizer()
        self.stemmer = PorterStemmer()
    def __call__(self, doc):
        return [self.stemmer.stem(w) for w in self.tokenizer(doc)]

def convert_to_tfidf(doc, min_df=2):
    '''
    Transform raw docs to TF-IDF feature vectors
    '''
    vectorizer = TfidfVectorizer(tokenizer=StemTokenizer(), stop_words='english', min_df=min_df)
    tfidf = vectorizer.fit_transform(doc)
    print("With min_df as {0}, features shape: {1}".format(min_df, tfidf.shape))
    return tfidf

def convert_to_tficf():
    '''
    Transform raw docs to TF-ICF feature vectors
    '''
    train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)

    class_text = [""]*20
    for i in range(len(train.data)):
        cat_index = train.target[i]
        class_text[cat_index] += (" "+train.data[i])
    tfidf_vectorizer = TfidfVectorizer(tokenizer=StemTokenizer(), stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(class_text)

    for i in range(20):
        if train.target_names[i] in mult_classes:
            cur_class = tfidf.toarray()[i]
            top_ten_index = sorted(range(len(cur_class)), key=lambda index: cur_class[index])[-10:]
            top_ten_term = []
            for index in top_ten_index:
                top_ten_term.append(tfidf_vectorizer.get_feature_names()[index])
            print(train.target_names[i])
            print(top_ten_term)

def apply_LSI(train):

    svd = TruncatedSVD(n_components=50, random_state=42)
    SVD_train = svd.fit_transform(convert_to_tfidf(train))
    print(SVD_train.shape)
    return SVD_train

def apply_NMF(train):
    model = NMF(n_components=50, init='random', random_state=42)
    W_train = model.fit_transform(convert_to_tfidf(train))
    print(W_train.shape)
    return W_train

def plot_roc(fpr, tpr):
    '''
    Plot of a ROC curve for a specific class
    '''
    roc_auc = auc(fpr, tpr)
    print('ROC AUC: %0.4f' % roc_auc)

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.grid(color='0.7', linestyle='--', linewidth=1)
    plt.xlim([-0.1, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.show()

def evaluate(y_true, y_pred, y_score, y_label, binary=True):
    '''
    Evaluation by given metrics
    '''
    if binary:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plot_roc(fpr, tpr)
    cnf_matrix = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cnf_matrix, y_label)
    print('\t-- accuracy score: %.3f' % metrics.accuracy_score(y_true, y_pred))
    if binary:
        print('\t-- recall score: %.3f' % metrics.recall_score(y_true, y_pred))
        print('\t-- precision score: %.3f' % metrics.precision_score(y_true, y_pred))
    else:
        print('\t-- recall score: %.3f' % metrics.recall_score(y_true, y_pred, average='macro'))
        print('\t-- precision score: %.3f' % metrics.precision_score(y_true, y_pred, average='macro'))

def main():
    args = sys.argv
    if len(args) != 2:
        print('Please select question part as "python newsgroup.py [index]"')
        return

    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
    # prepare data for e-h
    # [0 'comp.graphics',
    # 1   'comp.os.ms-windows.misc',
    # 2   'comp.sys.ibm.pc.hardware',
    # 3   'comp.sys.mac.hardware',
    # 4   'rec.autos',
    # 5   'rec.motorcycles',
    # 6   'rec.sport.baseball',
    # 7   'rec.sport.hockey']
    # 0-3 ---> 0
    # 4-7 ---> 1
    y_train = np.zeros(len(newsgroups_train.target))
    for i in range(len(y_train)):
        if newsgroups_train.target[i] >= 4:
            y_train[i] = 1
    y_test = np.zeros(len(newsgroups_test.target))
    for i in range(len(y_test)):
        if newsgroups_test.target[i] >= 4:
            y_test[i] = 1

    # task a
    # plot a histogram of the number of training documents
    # per class to check if they are evenly distributed
    if args[1] == 'a':
        plot_histogram(newsgroups_train)
    # task b
    elif args[1] == 'b':
        convert_to_tfidf(newsgroups_train.data, min_df=2)
        convert_to_tfidf(newsgroups_train.data, min_df=5)
    # task c
    elif args[1] == 'c':
        convert_to_tficf()
    # task d
    elif args[1] == 'd':
        apply_LSI(newsgroups_train.data)
        apply_NMF(newsgroups_train.data)
    # task e
    # binary SVM
    elif args[1] == 'e':
        pipe = Pipeline([('tfidf', TfidfVectorizer(tokenizer=StemTokenizer(), stop_words='english')),
                         ('lsi', TruncatedSVD(n_components=50, random_state=42)),
                         ('clf', svm.SVC(kernel='linear'))])
        param_grid = [{'clf__C': 1000, 'tfidf__min_df': 2},
                      {'clf__C': 1000, 'tfidf__min_df': 5},
                      {'clf__C': 0.001, 'tfidf__min_df': 2},
                      {'clf__C': 0.001, 'tfidf__min_df': 5}]
        for param in param_grid:
            print(param)
            pipe.set_params(**param)
            pipe.fit(newsgroups_train.data, y_train)
            y_score = pipe.decision_function(newsgroups_test.data)
            y_test_pred = pipe.predict(newsgroups_test.data)
            evaluate(y_test, y_test_pred, y_score, bin_classes)

    # task f
    # binary SVM CV
    elif args[1] == 'f':
        pipe = Pipeline([('tfidf', TfidfVectorizer(tokenizer=StemTokenizer(), stop_words='english', min_df=2)),
                         ('lsi', TruncatedSVD(n_components=50, random_state=42)),
                         ('clf', svm.SVC(kernel='linear'))])
        tuned_parameters = {'clf__C': [10**k for k in range(-3, 4)]}
        grid = GridSearchCV(pipe, tuned_parameters, cv=5)
        grid.fit(newsgroups_train.data, y_train)
        y_score = grid.decision_function(newsgroups_test.data)
        y_test_pred = grid.predict(newsgroups_test.data)
        evaluate(y_test, y_test_pred, y_score, bin_classes)

    # task g
    # binary naive bayes
    elif args[1] == 'g':
        pipe = Pipeline([('tfidf', TfidfVectorizer(tokenizer=StemTokenizer(), stop_words='english', min_df=5)),
                         ('nmf', NMF(n_components=50, init='random', random_state=42)),
                         ('clf', MultinomialNB())])
        pipe.fit(newsgroups_train.data, y_train)
        y_score = pipe.predict_proba(newsgroups_test.data)
        y_test_pred = pipe.predict(newsgroups_test.data)
        evaluate(y_test, y_test_pred, y_score[:, 1], bin_classes)

    # task h
    # binary logistic regression
    elif args[1] == 'h':
        pipe = Pipeline([('tfidf', TfidfVectorizer(tokenizer=StemTokenizer(), stop_words='english', min_df=5)),
                         ('nmf', NMF(n_components=50, init='random', random_state=42)),
                         ('clf', LogisticRegression())])
        pipe.fit(newsgroups_train.data, y_train)
        y_score = pipe.predict_proba(newsgroups_test.data)
        y_test_pred = pipe.predict(newsgroups_test.data)
        evaluate(y_test, y_test_pred, y_score[:, 1], bin_classes)

    # task i
    elif args[1] == 'i':
        pipe = Pipeline([('tfidf', TfidfVectorizer(tokenizer=StemTokenizer(), stop_words='english', min_df=5)),
                         ('nmf', NMF(n_components=50, init='random', random_state=42)),
                         ('clf', LogisticRegression())])
        tuned_parameters = {'clf__penalty': ['l1', 'l2'],
                            'clf__C': [10**k for k in range(-3, 4)]}
        grid = GridSearchCV(pipe, tuned_parameters)
        grid.fit(newsgroups_train.data, y_train)
        print(grid.best_params_)
        print(grid.cv_results_)

    # task j
    # multiclass
    elif args[1] == 'j':
        train_set = fetch_20newsgroups(subset='train', categories=mult_classes, shuffle=True, random_state=42)
        test_set = fetch_20newsgroups(subset='test', categories=mult_classes, shuffle=True, random_state=42)

        pipe = Pipeline([('tfidf', TfidfVectorizer(tokenizer=StemTokenizer(), stop_words='english', min_df=2)),
                         ('nmf', NMF(n_components=50, init='random', random_state=42)),
                         ('clf', LogisticRegression())])

        param_grid = [{'clf': MultinomialNB()},
                      {'clf': svm.SVC(kernel='linear')}, # The multiclass support of SVC is handled according to a one-vs-one scheme.
                      {'clf': OneVsRestClassifier(svm.SVC(kernel='linear'))}]
        for param in param_grid:
            print(param)
            pipe.set_params(**param)
            pipe.fit(train_set.data, train_set.target)
            if isinstance(param['clf'], MultinomialNB):
                y_score = pipe.predict_proba(test_set.data)
            else:
                y_score = pipe.decision_function(test_set.data)
            y_test_pred = pipe.predict(test_set.data)
            evaluate(test_set.target, y_test_pred, y_score, mult_classes, binary=False)

    else:
        print("undefined")

if __name__ == "__main__":
    main()
