from nltk.tokenize import sent_tokenize, word_tokenize 
import warnings 
import numpy as np
  
warnings.filterwarnings(action = 'ignore') 
  
import gensim 
from gensim.models import Word2Vec

from helpers import csv_to_df
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.svm import LinearSVC
from pprint import pprint
from time import time
import logging
from collections import defaultdict

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(next(iter(word2vec.values())))

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])


if __name__ == "__main__":
    
    train_data = csv_to_df('training.csv')
    test_data = csv_to_df('test.csv')

    X_train = train_data['article_words'].to_numpy()
    y_train = train_data['topic'].to_numpy()

    X_test = test_data['article_words'].to_numpy()
    y_test = test_data['topic'].to_numpy()

    with open("glove.6B.50d.txt", "rb") as lines:
        w2v = {line.split()[0]: np.array([float(num) for num in line.split()[1:]]) for line in lines}

    pipeline = Pipeline(
        [
            ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
            ("clf", BernoulliNB())
        ]
    )

    parameters = {
        #'vect__max_df': (0.5, 0.75, 1.0),
        #'vect__max_features': (None, 5000, 10000, 50000),
        #'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        #'tfidf__use_idf': (True, False),
        #'tfidf__norm': ('l1', 'l2'),
        #'clf__max_iter': (100,),
        #'clf__alpha': (0.00001, 0.000001),
        #'clf__penalty': ('l2', ),
        # 'clf__max_iter': (10, 50, 80),
    }

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, cv=5, refit=True)

    # print("Performing grid search...")
    # print("pipeline:", [name for name, _ in pipeline.steps])
    # print("parameters:")
    # pprint(parameters)
    # t0 = time()
    grid_search.fit(X_train, y_train)
    y_pred = grid_search.predict(X_test)
    print(accuracy_score(y_test, y_pred))

    # print("done in %0.3fs" % (time() - t0))
    # print()

    # print("Best estimator: ${0}".format(grid_search.best_estimator_))
    # print("Best score: %0.3f" % grid_search.best_score_)
    # print("Best parameters set:")
    # best_parameters = grid_search.best_estimator_.get_params()
    # for param_name in sorted(parameters.keys()):
    #     print("\t%s: %r" % (param_name, best_parameters[param_name]))



    

