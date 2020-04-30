import sys
import pandas as pd
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from helpers import csv_to_df
from nltk import FreqDist
from sklearn.linear_model import SGDClassifier
import xgboost as xgb
from sklearn.pipeline import Pipeline

def preprocess(train_data):

    all_words = []
    for article in train_text_data:
        article_words = list(set(article.split(',')))
        for word in article_words:
            all_words.append(word)

    dist = FreqDist(all_words)
    least_frequent_words = dist.hapaxes()
    for word in least_frequent_words:
        if(word in dist):
            del dist[word]

    vocab = set(dist.keys())
    return vocab

if __name__ == "__main__":

    train_data = csv_to_df('training.csv')
    train_text_data = train_data['article_words'].to_numpy()

    test_data = csv_to_df('test.csv')
    test_text_data = test_data['article_words'].to_numpy()

    vocab = preprocess(train_text_data)

    updated_train_data = pd.DataFrame(index=train_data.index.values.tolist(), columns=train_data.columns)
    train_count = 0
    for id, row, tag in zip(train_data['article_number'], train_data['article_words'], train_data['topic']):
        article_words = list(row.split(','))
        for word in article_words:
            if word not in vocab:
                article_words.remove(word)
        updated_train_data.loc[train_count , 'article_words'] = ','.join(article_words)
        updated_train_data.loc[train_count , 'topic'] = tag
        updated_train_data.loc[train_count , 'article_number'] = id
        train_count += 1

    updated_test_data = pd.DataFrame(index=test_data.index.values.tolist(), columns=test_data.columns)
    test_count = 0
    for id, row, tag in zip(test_data['article_number'], test_data['article_words'], test_data['topic']):
        article_words = list(row.split(','))
        for word in article_words:
            if word not in vocab:
                article_words.remove(word)
        updated_test_data.loc[test_count , 'article_words'] = ','.join(article_words)
        updated_test_data.loc[test_count, 'topic'] = tag
        updated_test_data.loc[test_count, 'article_number'] = id
        test_count += 1

    X_train = updated_train_data['article_words'].to_numpy()
    y_train = train_data['topic'].to_numpy()

    X_test = updated_test_data['article_words'].to_numpy()
    y_test = test_data['topic'].to_numpy()

    pipeline = Pipeline(
        [
            ('vect' , CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', xgb.XGBClassifier())
        ]
    )

    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        #'vect__max_features': (None, 5000, 10000, 50000),
        #'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        'clf__max_iter': (100,),
        #'clf__alpha': (0.00001, 0.000001),
        #'clf__penalty': ('l2', ),
        # 'clf__max_iter': (10, 50, 80),
    }
    
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, cv=10, refit=True)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    #pprint(parameters)
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
    

    # # Create bag of words
    # count = CountVectorizer()
    # bag_of_words = count.fit_transform(updated_train_text_data)
    # features = count.get_feature_names()
    
    # # Create feature matrix
    # X_train = bag_of_words

    # # Create target vector
    # y_train = updated_train_data['topic'].to_numpy()

    # clf = xgb.XGBClassifier()
    # model = clf.fit(X_train, y_train)

    # #test1 = count.transform(['heavy,heavy,gabriel,morn,morn,equit,cent,cent,cent,recent,stock,stock,stock,deficit,diversif,measur,rate,rate,rate,businessm,weekend,borrow,dharmal,take,term,affect,issu,worry,day,day,day,emerg,yesterday,play,friday,draw,million,million,million,money,point,point,point,light,amount,sold,baht,baht,baht,coupl,hold,pull,pull,time,busi,record,invest,governor,turn,bank,bank,bank,bank,tempor,tempor,billion,thai,thai,thai,thai,thai,hap,york,rang,account,account,overheat,econom,month,parallel,finish,fear,fear,percent,percent,ricard,peso,peso,peso,peso,anal,pick,wednesday,wednesday,holiday,advant,turnov,thailand,told,manuel,report,dollar,dollar,dollar,driv,driv,thing,settl,gosec,protect,protect,volum,current,execut,deal,deal,manil,manil,buy,overnight,secur,secur,monet,market,market,market,market,market,market,market,specul,concern,ebb,high,propert,cris,fact,philippin,philippin,philippin,philippin,interest,interest,de,call,trad,trad,trad,bought,clos,lift,presid,add,sell,mix,mix,start,devalu,long,author,dying,key,mid,vice,newsroom,jose,activ,sect,garcia,currenc,advis,compar,compar,singson,singson,singson,thursday,shed,sent,stop,arbitr'])
    # #test2 = count.transform(['world,world,nazarovi,medal,medal,end,braun,braun,braun,braun,point,point,point,heptathlon,heptathlon,race,athlet,strid,made,met,year,year,atlant,atlant,european,olymp,seventh,pass,finish,finish,finish,lead,lead,titl,titl,ahead,ahead,rival,disappoint,champ,champ,remig,recaptur,recaptur,bronz,bronz,denis,compet,german,monday,vict,lithuan,start,brit,total,sabin,lewi,lewi,lewi,lewi'])
    
    # X_test = count.transform(updated_test_text_data)
    # y_test = updated_test_data['topic'].to_numpy()

    # predicted_y = model.predict(X_test)
    # #predicted_probabilities = model.predict_proba(X_test)
    
    # print(accuracy_score(y_test , predicted_y))
    # print(classification_report(y_test , predicted_y , zero_division='warn'))