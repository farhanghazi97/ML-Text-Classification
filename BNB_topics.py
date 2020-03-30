import sys
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from helpers import csv_to_df, clean_df

if __name__ == "__main__":

    train_data = csv_to_df('training.csv')
    train_text_data = train_data['article_words'].to_numpy()

    test_data = csv_to_df('test.csv')
    test_text_data = test_data['article_words'].to_numpy()

    # Create bag of words
    count = CountVectorizer()
    bag_of_words = count.fit_transform(train_text_data)

    # Create feature matrix
    X_train = bag_of_words

    # Create target vector
    y_train = train_data['topic'].to_numpy()

    clf = BernoulliNB()
    model = clf.fit(X_train, y_train)

    #test1 = count.transform(['heavy,heavy,gabriel,morn,morn,equit,cent,cent,cent,recent,stock,stock,stock,deficit,diversif,measur,rate,rate,rate,businessm,weekend,borrow,dharmal,take,term,affect,issu,worry,day,day,day,emerg,yesterday,play,friday,draw,million,million,million,money,point,point,point,light,amount,sold,baht,baht,baht,coupl,hold,pull,pull,time,busi,record,invest,governor,turn,bank,bank,bank,bank,tempor,tempor,billion,thai,thai,thai,thai,thai,hap,york,rang,account,account,overheat,econom,month,parallel,finish,fear,fear,percent,percent,ricard,peso,peso,peso,peso,anal,pick,wednesday,wednesday,holiday,advant,turnov,thailand,told,manuel,report,dollar,dollar,dollar,driv,driv,thing,settl,gosec,protect,protect,volum,current,execut,deal,deal,manil,manil,buy,overnight,secur,secur,monet,market,market,market,market,market,market,market,specul,concern,ebb,high,propert,cris,fact,philippin,philippin,philippin,philippin,interest,interest,de,call,trad,trad,trad,bought,clos,lift,presid,add,sell,mix,mix,start,devalu,long,author,dying,key,mid,vice,newsroom,jose,activ,sect,garcia,currenc,advis,compar,compar,singson,singson,singson,thursday,shed,sent,stop,arbitr'])
    #test2 = count.transform(['world,world,nazarovi,medal,medal,end,braun,braun,braun,braun,point,point,point,heptathlon,heptathlon,race,athlet,strid,made,met,year,year,atlant,atlant,european,olymp,seventh,pass,finish,finish,finish,lead,lead,titl,titl,ahead,ahead,rival,disappoint,champ,champ,remig,recaptur,recaptur,bronz,bronz,denis,compet,german,monday,vict,lithuan,start,brit,total,sabin,lewi,lewi,lewi,lewi'])
    
    X_test = count.transform(test_text_data)
    y_test = test_data['topic'].to_numpy()

    predicted_y = model.predict(X_test)
    #predicted_probabilities = model.predict_proba(X_test)
    
    print(accuracy_score(y_test , predicted_y))
    print(classification_report(y_test , predicted_y , zero_division='warn'))
    


    

    
