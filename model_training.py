import sklearn.metrics as metrics
import matplotlib as mpl
from matplotlib import pyplot as plt
import math
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Dropout, Activation
import warnings
import numpy as np
import pandas as pd
from data_processing import data_processing
warnings.filterwarnings("ignore")

# ------------------------------------------------------This is a split line----


def __raw_lstm_model() -> Sequential:
    """_summary_

    Returns:
        Sequential: the raw LSTM model
    """

    raw_lstm_model = Sequential()

    # no need for the arg: units
    # It is just the numbers which are the sizes of LSTM layers
    # we can straightly set the numbers

    # Before input the train data into input layer, we need to reshape it first
    # This "12" in the input_shape is the "features" of X_train which is due to "lags"
    # If you wish to change the num of "lags", change the "12" here as well

    raw_lstm_model.add(LSTM(64, input_shape=(12, 1), return_sequences=True))
    raw_lstm_model.add(LSTM(64))
    raw_lstm_model.add(Dropout(rate=0.3))
    raw_lstm_model.add(Dense(1, activation='sigmoid'))

    return raw_lstm_model


def _lstm_training(X_train, y_train) -> Sequential:
    """training the model of LSTM

    Args:
        X_train (np.ndarray): the train dataset with shape
        y_train (np.ndarray): the train result with shape

    Returns:
        Sequential: the trained LSTM model
    """
    model = __raw_lstm_model()

    model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])

    model.fit(X_train, y_train, batch_size=256,
              epochs=600, validation_split=0.05)

    model.save("model/lstm_model_wtf.h5")

    return model


def __linear_regression():
    pass


def _linear_regression_training():
    pass


def model_training(train_path: str, test_path: str, lags: int) -> Sequential:
    """training all the models

    Args:
        train_path (str): the path of train dataset
        test_path (str): the path of test dataset, actually useless here but necessary
        lags (int): the number of lags which is window size for the flow data.
        That is to say how many days, you use their data to predict the future data

    Returns:
        Sequential: the trained model
    """

    X_train, y_train, _, _, _ = data_processing(train_path, test_path, lags)
    X_train_lstm = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = _lstm_training(X_train_lstm, y_train)

    return model


if __name__ == '__main__':

    train_path = "data/train.csv"
    test_path = "data/test.csv"

    lags = 13

    model = model_training(train_path, test_path, lags)
