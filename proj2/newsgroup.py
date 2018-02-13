import sys
import itertools
import matplotlib.pyplot as plt
import numpy as np
import nltk
import gc

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
from sklearn.decomposition import NMF
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix
from sklearn.decomposition import PCA
from sklearn import preprocessing

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

def evaluate(y_true, y_pred):
    '''
    Evaluation by given metrics
    '''
    # print(contingency_matrix(y_true, y_pred))
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(y_true, y_pred))
    print("Completeness: %0.3f" % metrics.completeness_score(y_true, y_pred))
    print("V-measure: %0.3f" % metrics.v_measure_score(y_true, y_pred))
    print("Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(y_true, y_pred))
    print("Adjusted Mutal Info: %.3f" % metrics.adjusted_mutual_info_score(y_true, y_pred))

def visualize(X, y, centers):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=50, alpha=0.5)
    plt.show()

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

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=3)
    tfidf = tfidf_vectorizer.fit_transform(newsgroups_train.data)

    if args[1] == '1':
        print(tfidf.shape)
    elif args[1] == '2':
        km = KMeans(n_clusters=2)
        km.fit(tfidf)
        evaluate(y_train, km.labels_)
    elif args[1] == '3':
        # part a
        svd = TruncatedSVD(n_components=1000, random_state=42)
        svd.fit(tfidf)
        ratio = [svd.explained_variance_ratio_[0]]
        for r in range(1, 1000):
            ratio.append(ratio[-1] + svd.explained_variance_ratio_[r])
        plt.plot(range(0, 1000), ratio)
        plt.xlabel('r')
        plt.ylabel('ratio of the variance')
        plt.show()

        # part b
        for r in [1, 2, 3, 5, 10, 20, 50, 100, 300]:
            print("="*20)
            print("For r = %d" % r)
            print("SVD")
            svd = TruncatedSVD(n_components=r, random_state=42)
            svd_train = svd.fit_transform(tfidf)
            km = KMeans(n_clusters=2)
            km.fit(svd_train)
            evaluate(y_train, km.labels_)
            print("NMF")
            nmf = NMF(n_components=r, init='random', random_state=42)
            nmf_train = nmf.fit_transform(tfidf)
            km = KMeans(n_clusters=2)
            km.fit(nmf_train)
            evaluate(y_train, km.labels_)
            gc.collect()
    elif args[1] == '4':
        # part a
        svd = TruncatedSVD(n_components=2, random_state=42)
        svd_train = svd.fit_transform(tfidf)
        km = KMeans(n_clusters=2)
        km.fit(svd_train)
        visualize(svd_train, km.labels_, km.cluster_centers_)

        # part b
        print("==unit variance==")
        X_scaled = preprocessing.scale(svd_train)
        del km
        km = KMeans(n_clusters=2)
        km.fit(X_scaled)
        evaluate(y_train, km.labels_)

        print("==logarithm transform==")
        nmf = NMF(n_components=50, init='random', random_state=42)
        nmf_train = nmf.fit_transform(tfidf)
        transformer = preprocessing.FunctionTransformer(np.log1p)
        log_trans = transformer.transform(nmf_train)
        del km
        km = KMeans(n_clusters=2)
        km.fit(log_trans)
        evaluate(y_train, km.labels_)

        print("==combine both==")
        nmf = NMF(n_components=50, init='random', random_state=42)
        nmf_train = nmf.fit_transform(tfidf)
        transformer = preprocessing.FunctionTransformer(np.log1p)

        print("unit variance first:")
        X_scaled = preprocessing.scale(nmf_train)
        log_trans = transformer.transform(X_scaled)
        del km
        km = KMeans(n_clusters=2)
        km.fit(log_trans)
        evaluate(y_train, km.labels_)

        print("log trans first:")
        log_trans = transformer.transform(nmf_train)
        X_scaled = preprocessing.scale(log_trans)
        del km
        km = KMeans(n_clusters=2)
        km.fit(X_scaled)
        evaluate(y_train, km.labels_)
    elif args[1] == '5':
        train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=3)
        tfidf = tfidf_vectorizer.fit_transform(train.data)

        print("SVD")
        svd = TruncatedSVD(n_components=20, random_state=42)
        svd_train = svd.fit_transform(tfidf)
        km = KMeans(n_clusters=20)
        km.fit(svd_train)
        evaluate(train.target, km.labels_)
        del km

        print("NMF")
        nmf = NMF(n_components=20, init='random', random_state=42)
        nmf_train = nmf.fit_transform(tfidf)
        km = KMeans(n_clusters=20)
        km.fit(nmf_train)
        evaluate(train.target, km.labels_)
        del km

        print("Normalized SVD")
        svd = TruncatedSVD(n_components=20, random_state=42)
        svd_train = svd.fit_transform(tfidf)
        X_scaled = preprocessing.scale(svd_train)
        km = KMeans(n_clusters=20)
        km.fit(X_scaled)
        evaluate(train.target, km.labels_)
        del km

        print("Logarithm transformed NMF")
        nmf = NMF(n_components=20, init='random', random_state=42)
        nmf_train = nmf.fit_transform(tfidf)
        transformer = preprocessing.FunctionTransformer(np.log1p)
        log_trans = transformer.transform(nmf_train)
        km = KMeans(n_clusters=20)
        km.fit(log_trans)
        evaluate(train.target, km.labels_)
        del km

        print("Normalized logarithm transformed NMF (Normalization first)")
        nmf = NMF(n_components=20, init='random', random_state=42)
        nmf_train = nmf.fit_transform(tfidf)
        X_scaled = preprocessing.scale(svd_train)
        transformer = preprocessing.FunctionTransformer(np.log1p)
        log_trans = transformer.transform(X_scaled)
        km = KMeans(n_clusters=20)
        km.fit(log_trans)
        evaluate(train.target, km.labels_)
        del km

        print("Normalized logarithm transformed NMF (Logarithm transformation first)")
        nmf = NMF(n_components=20, init='random', random_state=42)
        nmf_train = nmf.fit_transform(tfidf)
        transformer = preprocessing.FunctionTransformer(np.log1p)
        log_trans = transformer.transform(nmf_train)
        X_scaled = preprocessing.scale(log_trans)
        km = KMeans(n_clusters=20)
        km.fit(X_scaled)
        evaluate(train.target, km.labels_)
    else:
        print("undefined")

if __name__ == "__main__":
    main()
