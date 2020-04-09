from helpers import csv_to_df
import datetime
import json
import os
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import time
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import Sequential
from keras.layers import Embedding, SpatialDropout1D, LSTM, Dense
from keras.callbacks.callbacks import EarlyStopping

if __name__ == "__main__":

    training_set = csv_to_df('training.csv')
    test_set = csv_to_df('test.csv')
    
    MAX_NB_WORDS = 50000
    MAX_SEQUENCE_LENGTH = 500
    EMBEDDING_DIM = 100
    
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, split=',')
    tokenizer.fit_on_texts(training_set['article_words'].values)
    word_index = tokenizer.word_index
    #print('Found %s unique tokens.' % len(word_index))

    X_train = tokenizer.texts_to_sequences(training_set['article_words'].values)
    X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
    #print('Shape of data tensor:', X.shape)

    y_train = pd.get_dummies(training_set['topic']).values
    #print('Shape of label tensor:', Y.shape)
    
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X_train.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.4, recurrent_dropout=0.2))
    model.add(Dense(11, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    epochs = 1
    batch_size = 32

    history = model.fit(
        X_train, y_train, 
        epochs=epochs, 
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)]
    )

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, split=',')
    tokenizer.fit_on_texts(test_set['article_words'].values)
    word_index = tokenizer.word_index

    X_test = tokenizer.texts_to_sequences(test_set['article_words'].values)
    X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)

    y_test = pd.get_dummies(test_set['topic']).values

    topic_data = test_set.topic.value_counts()

    accr = model.evaluate(X_test, y_test)
    y_pred = model.predict(X_test)
    print(y_pred.shape)
    labels = list(topic_data.keys())
    print(y_pred, labels[np.argmax(y_pred)])
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
    


    
        

    
        