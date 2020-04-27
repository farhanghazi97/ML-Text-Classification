import pandas as pd
import numpy as np
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('training.csv')
test = pd.read_csv('test.csv')
combined = pd.concat([train,test], axis=0)

y_train = train['topic']
y_test = test['topic']

Tfidf_vect = TfidfVectorizer(max_features=2000,strip_accents='ascii')
Tfidf_vect.fit(combined['article_words'])

X_train = Tfidf_vect.transform(train['article_words'])
X_test = Tfidf_vect.transform(test['article_words'])

SVM = svm.SVC(C=1.0, kernel='linear', gamma='auto', probability=True)
SVM.fit(X_train,y_train)
predictions_SVM = SVM.predict(X_test)
print("Score: ",accuracy_score(predictions_SVM, y_test)*100)