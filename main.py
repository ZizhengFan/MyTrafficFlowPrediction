from timeit import timeit
import warnings
import numpy as np
import pandas as pd
from keras.models import load_model
warnings.filterwarnings("ignore")

from data_processing import data_processing
from result_metrics import eva_regress, plot_results
from model_training import model_training
# ------------------------------------------------------This is a split line----

def predict_and_evaluate(train_path, test_path, lags):
    """from the trained model and test dataset, predict the results and evaluate

    Args:
        train_path (str): the path of train dataset
        test_path (str): the path of test dataset, actually useless here but necessary
        lags (int): the number of lags which is window size for the flow data.
        That is to say how many days, you use their data to predict the future data
    """
    
    # Here to determine if you want to retrain the model or just use the existing one
    # lstm, regression = model_training(train_path, test_path, lags)
    
    lstm = load_model('model/lstm_model_correct.h5')
    regression = load_model('model/regression_model.h5')
    
    y_lstm_predict, y_regression_predict= [], []
    
    _, _, X_test, y_test, scaler = data_processing(train_path, test_path, lags)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]
    
    lstm_predicted = lstm.predict(X_test)
    lstm_predicted = scaler.inverse_transform(lstm_predicted.reshape(-1, 1)).reshape(1, -1)[0]

    y_lstm_predict.append(lstm_predicted[288:288+289])
    eva_regress(y_test, lstm_predicted)
    
    plot_results(y_test[288:288+289], y_lstm_predict, ['LSTM'])
    
    regression_predicted = lstm.predict(X_test)
    regression_predicted = scaler.inverse_transform(regression_predicted.reshape(-1, 1)).reshape(1, -1)[0]

    y_regression_predict.append(regression_predicted[288+289:288+289+289])
    eva_regress(y_test, regression_predicted)
    
    plot_results(y_test[288+289:288+289+289], y_regression_predict, ['REGRESSION'])
    
    
def main():
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'
    lags = 13
    predict_and_evaluate(train_path, test_path, lags)
    

if __name__ == '__main__':
    main()
    