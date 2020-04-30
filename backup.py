import sys
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.svm import LinearSVC
from helpers import csv_to_df
from pprint import pprint
from time import time
import logging

if __name__ == "__main__":

    train_data = csv_to_df('training.csv')

    for row in train_data['article_words']:
        processed_words = row.split(',')
        updated_words = [word for word in processed_words if len(word) != 2]
        str_words = ','.join(updated_words)
        row = str_words

    test_data = csv_to_df('test.csv')
    test_text_data = test_data['article_words'].to_numpy()

    train_text_data = train_data['article_words'].to_numpy()

    X_train = train_text_data
    y_train = train_data['topic'].to_numpy()

    X_test = test_text_data
    y_test = test_data['topic'].to_numpy()

    # Define a pipeline combining a text feature extractor with a simple
    # classifier
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier()),
    ])

    # uncommenting more parameters will give better exploring power but will
    # increase processing time in a combinatorial way
    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        #'vect__max_features': (None, 5000, 10000, 50000),
        #'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        'clf__max_iter': (100,),
        #'clf__alpha': (0.00001, 0.000001),
        'clf__penalty': ('l2', ),
        # 'clf__max_iter': (10, 50, 80),
    }

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, cv=10, refit=True)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(X_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best estimator: ${0}".format(grid_search.best_estimator_))
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))