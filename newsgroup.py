import matplotlib.pyplot as plt
import numpy as np
import nltk
import string

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
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression


from scipy import interp
from sets import Set
def plot_histogram(newsgroups_train):
    '''
    Plot the document size per class histogram
    '''
    #class_size: the document size of each class,
    class_size = [0]*8
    for target in newsgroups_train.target:
        class_size[target]+=1
    #print class_size
    #[584, 591, 590, 578, 594, 598, 597, 600]

    n_groups = 8;            
    fig, ax = plt.subplots()    
    index = np.arange(n_groups)    
    bar_width = 0.35    
    opacity = 0.4    
    rects = plt.bar(index, class_size, bar_width,alpha=opacity, color='b')    
       
    plt.xlabel('Class')    
    plt.ylabel('Number of Documnet')    
    plt.title('Number of training document per class')    
    plt.xticks(index, ('0', '1', '2', '3', '4','5','6','7'))    
    plt.ylim(0,650);    
    plt.legend();    
      
    plt.tight_layout();   
    plt.show();    

def stem_tokens(tokens):
    stemmer = PorterStemmer()
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    '''
    override the tokenizer in CountVectorizer 
    with stemming and reomoving punctuation
    '''
    tokenizer = RegexpTokenizer(r'[A-Za-z_]+') #remove punctuation
    tokens = tokenizer.tokenize(text)
    stems = stem_tokens(tokens)
    return stems


def convert_to_tfidf(train):

    # tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words='english',min_df=5)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english',min_df=5)
    tfidf = tfidf_vectorizer.fit_transform(train.data)
    #print train_counts.shape
    print tfidf.shape
    return tfidf

def convert_to_tficf():
    train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
   
    class_text = [""]*20
    for i in range(len(train.data)):
        cat_index = train.target[i]
        class_text[cat_index] += (" "+train.data[i])
    # tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(class_text)

    classes = Set(['comp.sys.ibm.pc.hardware', 
    'comp.sys.mac.hardware', 'misc.forsale',
     'soc.religion.christian'])

    for i in range (20):
        if train.target_names[i] in classes:
            cur_class = tfidf.toarray()[i]
            top_ten_index = sorted(range(len(cur_class)), key=lambda index: cur_class[index])[-10:]
            top_ten_term = []
            for index in top_ten_index:
                top_ten_term.append(tfidf_vectorizer.get_feature_names()[index])
            print(train.target_names[i])
            print(top_ten_term)

def apply_lsi(train):

    svd = TruncatedSVD(n_components=50, random_state=42)
    SVD_train = svd.fit_transform(convert_to_tfidf(train))
    print SVD_train.shape
    return SVD_train

def apply_NMF(train):
    model = NMF(n_components=50, init='random', random_state=42)
    W_train = model.fit_transform(convert_to_tfidf(train))
    return W_train

def SVM_classifier(X,y,gam):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    clf = svm.SVC(kernel='linear', C=gam).fit(X_train,y_train)

    y_test_pred = clf.predict(X_test)
       
    print('\t-- accuracy score: %.3f' % metrics.accuracy_score(y_test, y_test_pred) )
    print('\t-- recall score: %.3f' % metrics.recall_score(y_test, y_test_pred) )
    print('\t-- precision score: %.3f' % metrics.precision_score(y_test, y_test_pred) )
    print metrics.confusion_matrix(y_test, y_test_pred)

    # fpr, tpr, thresholds = roc_curve(y_test, y_test_pred,pos_label = 2)
    # roc_auc = auc(fpr, tpr)  
    
    # print 'ROC AUC: %0.2f' % roc_auc

    # # Plot of a ROC curve for a specific class
    # plt.figure()
    # plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve')
    # plt.legend(loc="lower right")
    # plt.show()


def SVM_cross_val(X,y,gam):
    clf = svm.SVC(kernel='linear', C=gam)
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print  '%f  %f '%(gam,scores.mean())


def NB_classifier(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = MultinomialNB().fit(X_train,y_train)
    y_test_pred = clf.predict(X_test)
       
    print('\t-- accuracy score: %.3f' % metrics.accuracy_score(y_test, y_test_pred) )
    print('\t-- recall score: %.3f' % metrics.recall_score(y_test, y_test_pred) )
    print('\t-- precision score: %.3f' % metrics.precision_score(y_test, y_test_pred) )
    print metrics.confusion_matrix(y_test, y_test_pred)

def Log_classifier(X,y,norm='l2',regular_para=1.0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # clf = LogisticRegression().fit(X_train,y_train)
    clf = LogisticRegression(penalty=norm,C=regular_para).fit(X_train,y_train)
    y_test_pred = clf.predict(X_test)
       
    print('\t-- accuracy score: %.3f' % metrics.accuracy_score(y_test, y_test_pred) )
    print('\t-- recall score: %.3f' % metrics.recall_score(y_test, y_test_pred) )
    print('\t-- precision score: %.3f' % metrics.precision_score(y_test, y_test_pred) )
    print metrics.confusion_matrix(y_test, y_test_pred)



def main():
    categories = ['comp.graphics', 
    'comp.os.ms-windows.misc', 
    'comp.sys.ibm.pc.hardware', 
    'comp.sys.mac.hardware', 
    'rec.autos', 
    'rec.motorcycles', 
    'rec.sport.baseball', 
    'rec.sport.hockey']
    newsgroups_train = fetch_20newsgroups(subset='train', 
    categories=categories, shuffle=True, random_state=42)
    # print newsgroups_train.target.shape
    '''
    task a
     plot a histogram of the number of training documents 
     per class to check if they are evenly distributed
    '''
    #plot_histogram(newsgroups_train)
    '''
     task b
    '''
    # convert_to_tfidf(newsgroups_train)
    '''
    task c
    '''
    # convert_to_tficf()
    '''
    task d
    '''
    # apply_lsi(newsgroups_train)
    # apply_NMF(newsgroups_train)
    '''
    prepare data for e-h

    [0 'comp.graphics',   
    1   'comp.os.ms-windows.misc', 
    2   'comp.sys.ibm.pc.hardware', 
    3   'comp.sys.mac.hardware',
    4   'rec.autos', 
    5   'rec.motorcycles', 
    6   'rec.sport.baseball', 
    7   'rec.sport.hockey']

    0-3 ---> 0
    4-7 ---> 1
    '''

    y = np.zeros(len(newsgroups_train.data))
    for i in range (len(newsgroups_train.data)):
        if newsgroups_train.target[i]<=3:
            y[i]=0
        else:
            y[i]=1
    # print y

    # X = apply_lsi(newsgroups_train)
    X = apply_NMF(newsgroups_train)
    # X = convert_to_tfidf(newsgroups_train)
    print X.shape

    '''
    task e
    '''
    # SVM_classifier(X,y,1000)
    # SVM_classifier(X,y,0.001)
    

    # fpr, tpr, thresholds = roc_curve(y_test, y_test_pred,pos_label = 2)
    # roc_auc = auc(fpr, tpr)  
    
    # print 'ROC AUC: %0.2f' % roc_auc

    # # Plot of a ROC curve for a specific class
    # plt.figure()
    # plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve')
    # plt.legend(loc="lower right")
    # plt.show()

    '''
    task f
    '''
    # for i in range (-3,4):
    #     SVM_cross_val(X,y,pow(10,i))

    '''
    task g
    '''
    # NB_classifier(X,y)

    '''
    task h,i
    '''
    for i in range (1,16):
        print ("-------%d---------"%i)
        Log_classifier(X,y,'l1',i)
    
    


if __name__ == "__main__":
    main()
      