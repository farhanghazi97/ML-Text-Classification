import sys
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from helpers import csv_to_df

if __name__ == "__main__":

    train_data = csv_to_df('training.csv')
    train_text_data = train_data['article_words'].to_numpy()

    test_data = csv_to_df('test.csv')
    test_text_data = test_data['article_words'].to_numpy()

    # Create bag of words
    # count = CountVectorizer(stop_words='english')
    # bag_of_words = count.fit_transform(train_text_data)

    # Create feature matrix
    X_train = train_text_data

    # Create target vector
    y_train = train_data['topic'].to_numpy()

    X_test = test_text_data

    y_test = test_data['topic'].to_numpy()

    sgd = Pipeline(
        [
            ('vect', CountVectorizer()),
            ('tf-idf' , TfidfTransformer()),
            ('SGD', SGDClassifier(loss='hinge', penalty='l1',alpha=1e-3, random_state=42, tol=None)),
        ]
    )

    bnb = Pipeline(
        [
            ('vect', CountVectorizer()),
            ('tf-idf' , TfidfTransformer()),
            ('BNB', BernoulliNB()),
        ]
    )

    mnb = Pipeline(
        [
            ('vect', CountVectorizer()),
            ('tf-idf' , TfidfTransformer()),
            ('MNB', MultinomialNB()),
        ]
    )

    rf = Pipeline(
        [
            ('vect', CountVectorizer()),
            ('tf-idf' , TfidfTransformer()),
            ('random_forest', RandomForestClassifier()),
        ]
    )

    pipelines = [rf, bnb, mnb, sgd]

    pipe_dict = {0 : 'SGD' , 1 : 'BNB', 2 : 'MNB', 3 : 'RF'}

    for pipe in pipelines:
        pipe.fit(X_train, y_train)

    for i , model in enumerate(pipelines):
        predicted_y = model.predict(X_test)
        print("{0} - {1}".format(pipe_dict[i] , classification_report(y_test , predicted_y)))