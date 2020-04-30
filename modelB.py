from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from helpers import csv_to_df
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
import build_model
import numpy as np

# Vectorization parameters
# Range (inclusive) of n-gram sizes for tokenizing text.
NGRAM_RANGE = (1, 2)

# Limit on the number of features. We use the top 20K features.
TOP_K = 2000

# Whether text should be split into word or character n-grams.
# One of 'word', 'char'.
TOKEN_MODE = 'word'

# Minimum document/corpus frequency below which a token will be discarded.
MIN_DOCUMENT_FREQUENCY = 120

def ngram_vectorize(train_texts, train_labels, val_texts):
    """Vectorizes texts as n-gram vectors.

    1 text = 1 tf-idf vector the length of vocabulary of unigrams + bigrams.

    # Arguments
        train_texts: list, training text strings.
        train_labels: np.ndarray, training labels.
        val_texts: list, validation text strings.

    # Returns
        x_train, x_val: vectorized training and validation texts
    """
    # Create keyword arguments to pass to the 'tf-idf' vectorizer.
    kwargs = {
            'ngram_range': NGRAM_RANGE,  # Use 1-grams + 2-grams.
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': TOKEN_MODE,  # Split text into word tokens.
            'min_df': MIN_DOCUMENT_FREQUENCY,
            'tokenizer': lambda x:x.split(','),
            
    }
    vectorizer = TfidfVectorizer(**kwargs)

    # Learn vocabulary from training texts and vectorize training texts.
    x_train = vectorizer.fit_transform(train_texts)

    # Vectorize validation texts.
    x_val = vectorizer.transform(val_texts)

    # Select top 'k' of the vectorized features.
    selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train).astype('float32')
    x_val = selector.transform(x_val).astype('float32')
    return x_train, x_val, vectorizer

def ngram_vectorize_test_data(test_texts, test_labels, vectorizer):

    # Learn vocabulary from training texts and vectorize training texts.
    X_test = vectorizer.transform(test_texts)

    # Select top 'k' of the vectorized features.
    selector = SelectKBest(f_classif, k=min(TOP_K, X_test.shape[1]))
    selector.fit(X_test, test_labels)
    X_test = selector.transform(X_test).astype('float32')
    return X_test

if __name__ == "__main__":

    training_data = csv_to_df('training.csv')
    training_data = training_data.set_index('article_number')

    X_train, X_val, y_train, y_val = train_test_split(training_data['article_words'], training_data['topic'], test_size=0.10, random_state=42)
    X_train, X_val, vectorizer = ngram_vectorize(X_train, y_train, X_val)

    LB = LabelEncoder()
    train_labels = LB.fit_transform(y_train)
    val_labels = LB.fit_transform(y_val)

    num_classes = len(list((training_data['topic'].value_counts()).keys()))

    model = build_model.mlp_model(
        layers=2,
        units=32,
        dropout_rate=0.2,
        input_shape=X_train.shape[1:],
        num_classes=num_classes
    )

    # Compile model with learning parameters.
    if num_classes == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'
    optimizer = Adam(lr=1e-3)
    model.compile(optimizer='adam', loss=loss, metrics=['acc'])

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    callbacks = [EarlyStopping(monitor='val_loss', patience=2)]

    # Train and validate model.
    history = model.fit(
        X_train,
        train_labels,
        epochs=50,
        callbacks=callbacks,
        validation_data=(X_val, val_labels),
        verbose=2,  # Logs once per epoch.
        batch_size=64
    )

    # Print results.
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    # Read in test file
    test_data = csv_to_df('test.csv')
    
    # Get test data
    X_test = test_data['article_words']
    y_test = test_data['topic']
    article_ID = test_data['article_number']

    test_data = test_data.astype({'article_number' : 'int32'})
    test_data = test_data.set_index('article_number')
    
    # Preprocess test data
    X_test = ngram_vectorize_test_data(X_test, y_test, vectorizer)

    # Label encode test data labels
    test_labels = LB.transform(y_test)

    # Make prediction on test set
    y_pred = model.predict(X_test)

    # Evalulate model performance on test set
    score, acc = model.evaluate(X_test, test_labels, batch_size=64)
    print("Loss score: {0}, Accuracy: {1}".format(score, acc))

    # Create dict to contain predicted labels for each article instance
    topic_wise_labels = {}
    for topic in LB.classes_:
        topic_wise_labels[topic] = []
    topic_wise_pred_labels = {}
    for topic in LB.classes_:
        topic_wise_pred_labels[topic] = []

    # Update topic dict and create pred_list for CR
    full_set_pred_labels = []
    for ID, actual_label, model_pred in zip(article_ID, y_test , y_pred):
        prediction = LB.classes_[np.argmax(model_pred)]
        full_set_pred_labels.append(prediction)
        topic_wise_labels[prediction].append((int(ID) , model_pred[np.argmax(model_pred)]))

    # Sort each topic list by score value
    for topic in topic_wise_labels:
        topic_wise_labels[topic] = sorted(topic_wise_labels[topic] , key=lambda x: x[1], reverse=True)

    # Consolidate actual labels into dict of structure {'topic1' : [list of labels] , 'topic2': [list of labels]}
    for topic in topic_wise_labels:
        for topic_list in topic_wise_labels[topic]:
            topic_wise_pred_labels[topic].append(topic)
    actual_label_dict = {}
    for topic in topic_wise_labels:
        lst = [tup[0] for tup in topic_wise_labels[topic]]
        rows = test_data.loc[lst , 'topic'].to_list()
        actual_label_dict[topic] = rows

    # Run classification report for each topic
    for (Atopic, Alabel), (Ptopic, Plabel) in zip(actual_label_dict.items(), topic_wise_pred_labels.items()):
        print("----------{0}----------".format(Atopic))
        if(len(topic_wise_pred_labels[Atopic]) > 0):
            print(classification_report(actual_label_dict[Atopic] , topic_wise_pred_labels[Ptopic], zero_division=0))
        else:
            print("No predictions made for this topic!")

    # Calculate metrics via classification report
    print(classification_report(y_test, full_set_pred_labels, zero_division=0))