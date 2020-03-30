import sys
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.svm import LinearSVC
from helpers import csv_to_df

if __name__ == "__main__":

    train_data = csv_to_df('training.csv')
    train_text_data = train_data['article_words'].to_numpy()

    test_data = csv_to_df('test.csv')
    test_text_data = test_data['article_words'].to_numpy()

    # Create feature matrix
    X_train = train_text_data

    # Create target vector
    y_train = train_data['topic'].to_numpy()

    X_test = test_text_data

    y_test = test_data['topic'].to_numpy()

    LSVC = Pipeline(
        [
            ('vect', TfidfVectorizer(stop_words='english')),
            ('LinearSVC', LinearSVC()),
        ]
    )

    sgd = Pipeline(
        [
            ('vect', TfidfVectorizer(stop_words='english')),
            ('SGD', SGDClassifier()),
        ]
    )

    bnb = Pipeline(
        [
            ('vect', TfidfVectorizer(stop_words='english')),
            ('BNB', BernoulliNB(fit_prior=True)),
        ]
    )

    mnb = Pipeline(
        [
            ('vect', TfidfVectorizer(stop_words='english')),
            ('MNB', MultinomialNB(fit_prior=True)),
        ]
    )

    rf = Pipeline(
        [
            ('vect', TfidfVectorizer(stop_words='english')),
            ('random_forest', RandomForestClassifier()),
        ]
    )

    pipelines = [LSVC, sgd, bnb, mnb, rf]

    pipe_dict = {0 : 'LSVC' , 1 : 'SGD', 2 : 'BNB', 3 : 'MNB', 4 : 'RF'}

    for pipe in pipelines:
        pipe.fit(X_train, y_train)

    for i , model in enumerate(pipelines):
        predicted_y = model.predict(X_test)
        print("{0} - {1}".format(pipe_dict[i] , classification_report(y_test , predicted_y , zero_division=0)))