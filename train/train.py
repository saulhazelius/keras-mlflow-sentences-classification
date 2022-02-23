#!/usr/bin/env python
# coding: utf-8

import sys
from urllib.parse import urlparse
import logging
import pandas as pd
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import mlflow
import mlflow.keras


mlflow.keras.autolog()

logging.basicConfig(level=logging.INFO)
nltk.download("wordnet")
nltk.download("omw-1.4")


def get_processed_text(df, tokenizer, stopwords):

    processed_tweets = []  # list of tweeter tokens lists from text

    for text in df.text:
        tokens = tokenizer.tokenize(text)  #
        processed_tweets.append(
            " ".join(
                [
                    lemmatizer.lemmatize(token)
                    for token in tokens
                    if token not in stopwords
                    and not token.isdigit()
                    and token not in punctuation
                    and not "http" in token  #
                ]
            )
        )

    return processed_tweets

if __name__ == '__main__':
    

    df_train = pd.read_csv("../data/train.csv")
    df_test = pd.read_csv("../data/test.csv")

    stopws = set(stopwords.words("english"))
    tk = TweetTokenizer(strip_handles=True, preserve_case=False)
    lemmatizer = WordNetLemmatizer()


    corpus_train = get_processed_text(df_train, tk, stopws)
    corpus_test = get_processed_text(df_test, tk, stopws)

    # ### Bi-LSTM

    ## Model:

    
    hid_units = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    dp = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1

    inputs = keras.Input(shape=(None,), dtype="int32")
    x = layers.Embedding(5000, hid_units)(inputs)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=dp))(x)
    x = layers.Bidirectional(layers.LSTM(32))(x)
    # Classifier:
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        "adam",
        "binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC(),
        ],
    )


    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(corpus_train)

    X_train = tokenizer.texts_to_sequences(corpus_train)
    X_test = tokenizer.texts_to_sequences(corpus_test)

    X_train = pad_sequences(X_train, padding="post", maxlen=30)
    X_test = pad_sequences(X_test, padding="post", maxlen=30)

    y_train = df_train.target.values
    y_test = df_test.target.values

    logging.info('Training...')
    model.fit(X_train, y_train, epochs=2)

    print(f" Accuracy: {model.evaluate(X_test, y_test)[0]}")
    print(f" Precision: {model.evaluate(X_test, y_test)[1]}")
    print(f" Recall: {model.evaluate(X_test, y_test)[2]}")
    print(f" ROC AUC: {model.evaluate(X_test, y_test)[3]}")

    mlflow.log_param("hidden-units", hid_units)
    mlflow.log_param("dropout", dp)
    mlflow.log_metric("accuracy", model.evaluate(X_test, y_test)[0])
    mlflow.log_metric("precision", model.evaluate(X_test, y_test)[1])
    mlflow.log_metric("recall", model.evaluate(X_test, y_test)[2])
    mlflow.log_metric("roc-auc", model.evaluate(X_test, y_test)[3])

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # Model registry does not work with file store
    if tracking_url_type_store != "file":

        # Register the model
        # There are other ways to use the Model Registry, which depends on the use case,
        # please refer to the doc for more information:
        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.keras.log_model(model, "model", registered_model_name="KerasTweets")
    else:
        mlflow.keras.log_model(model, "model")

    
