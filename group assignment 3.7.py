#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
training = pd.read_csv('training.csv')
test = pd.read_csv('test.csv')
training.head()


# In[2]:


from io import StringIO
col = ['topic', 'article_words']
training = training[col]
training[pd.notnull(training['article_words'])]

training.columns = ['topic', 'article_words']

training['category_id'] = training['topic'].factorize()[0]
category_id_training = training[['topic', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_training.values)
id_to_category = dict(category_id_training[['category_id', 'topic']].values)
training.head()


# In[3]:


import matplotlib.pyplot as plt
fig = plt.figure(figsize = (8,6))
training.groupby('topic').article_words.count().plot.bar(ylim=0)
plt.show()


# In[4]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(sublinear_tf = True, min_df=5, encoding='latin-1', ngram_range=(1,2), stop_words='english')

features = tfidf.fit_transform(training.article_words).toarray()
labels = training.category_id
features.shape


# In[5]:


from sklearn.feature_selection import chi2
import numpy as np

N=2
for topic, category_id in sorted(category_to_id.items()):
    features_chi2 = chi2(features, labels == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("# '{}':".format(topic))
    print("  .Most correlated unigrams:\n. {}".format('\n.'.join(unigrams[-N:])))
    print("  .Most correlated bigrams:\n. {}".format('\n.'.join(bigrams[-N:])))


# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split(training['article_words'], training['topic'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, y_train)


# In[7]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score

models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(), 
    MultinomialNB(), 
    LogisticRegression(random_state=0)
]
CV = 5
cv_training = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
cv_training = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

import seaborn as sns

sns.boxplot(x='model_name', y='accuracy', data=cv_training)
sns.stripplot(x='model_name', y='accuracy', data=cv_training, size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()


# In[8]:


cv_training.groupby('model_name').accuracy.mean()


# In[9]:


model = LinearSVC()

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, training.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=category_id_training.topic.values,yticklabels=category_id_training.topic.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


# In[10]:


from IPython.display import display

for predicted in category_id_training.category_id:
    for actual in category_id_training.category_id:
        if predicted != actual and conf_mat[actual, predicted] > 10:
            print("'{}' predicted as '{}' : {} examples.".format(id_to_category[actual], id_to_category[predicted], conf_mat[actual, predicted]))
            display(training.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['topic', 'article_words']])
            print('')


# In[12]:


model.fit(features, labels)

N = 2
for topic, category_id in sorted(category_to_id.items()):
    indices = np.argsort(model.coef_[category_id])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
    bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
    print("# '{}':".format(topic))
    print("  . Top unigrams:\n .{}".format('\n. '.join(unigrams)))
    print("  . Top bigrams:\n .{}".format('\n. '.join(bigrams)))


# In[13]:


from sklearn import metrics
print(metrics.classification_report(y_test, y_pred, target_names=training['topic'].unique()))


# In[ ]:




