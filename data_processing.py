import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# ------------------------------------------------------This is a split line----


def _data_reading(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """reading data from path

    Args:
        train_path (str): the path of train.csv
        test_path (str): the path of test.csv

    Returns:
        pd.DataFrame: train data as pandas dataframe with correct columns
        pd.DataFrame: test data as pandas dataframe with correct columns
    """
    df_train = pd.read_csv(train_path, encoding="utf-8")
    df_test = pd.read_csv(test_path, encoding="utf-8")

    df_train.columns = ["utc_time", "flow", "lane_id", "observe"]
    df_test.columns = ["utc_time", "flow", "lane_id", "observe"]

    # print(df_train.columns)
    return df_train, df_test


def _data_checking(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """check if there is missing, duplicated and outlier data

    Args:
        df_train (pd.DataFrame): train data as pandas dataframe with correct columns
        df_test (pd.DataFrame): test data as pandas dataframe with correct columns
    """
    print(df_train.isnull().sum())
    print(df_test.isnull().sum())
    print(df_train.duplicated().sum())
    print(df_test.duplicated().sum())
    # print(df_train.describe())
    # print(df_test.describe())

    # TODO: Maybe here will be missing value handler or something


def _data_scaler(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple[MinMaxScaler,
                                                                         np.ndarray,
                                                                         np.ndarray]:
    """scaling data into [0, 1]

    Args:
        df_train (_type_): train data as pandas dataframe with correct columns
        df_test (_type_): test data as pandas dataframe with correct columns

    Returns:
        np.ndarray: the scaled feature "flow" of train data
        np.ndarray: the scaled feature "flow" of test data
        MinMaxScaler: the scaler used to scale data

    """

    # initialize the scaler which fit the "flow" feature of df_train
    # .values is to convert the pandas series to numpy array
    # .reshape is to change the shape of the array into one column
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(
        df_train["flow"].values.reshape(-1, 1))

    # use the scaler to scale the "flow" feature of df_train and df_test
    # .reshape(1, -1) is to change the shape of the array into one row
    # [0] is to reduce the demension of the array
    flow_train_raw = scaler.transform(
        df_train["flow"].values.reshape(-1, 1)).reshape(1, -1)[0]
    flow_test_raw = scaler.transform(
        df_test["flow"].values.reshape(-1, 1)).reshape(1, -1)[0]

    return flow_train_raw, flow_test_raw, scaler


def _traintest_generating(flow_train_raw, flow_test_raw, lags: int) -> tuple[np.ndarray,
                                                                             np.ndarray,
                                                                             np.ndarray,
                                                                             np.ndarray]:
    """generating train dataset and test dataset with lags

    Args:
        flow_train (np.ndarray): the scaled feature "flow" of train data
        flow_test (np.ndarray): the scaled feature "flow" of test data
        lags (int): the number of lags which is window size for the flow data.
        That is to say how many days, you use their data to predict the future data

    Returns:
        np.ndarray: the X_train data which is the (lag-1) days each row
        np.ndarray: the y_train data which is the day after (lag-1) days each row
        np.ndarray: the X_test data which is the (lag-1) days each row
        np.ndarray: the y_test data which is the day after (lag-1) days each row
    """

    # Here is to generate the train and test dateset where the data sliding window
    # is "lags" days and slide one day each time
    # which means we will use the data from "lag - 1" days to predict the future data
    # the X_train format will be like:
    # 1 2 3 4 5
    # 2 3 4 5 6
    # 3 4 5 6 7
    # ...
    flow_train, flow_test = [], []

    for i in range(0, len(flow_train_raw)-lags):
        flow_train.append(flow_train_raw[i:i+lags])
    for i in range(0, len(flow_test_raw)-lags):
        flow_test.append(flow_test_raw[i:i+lags])

    flow_train = np.array(flow_train)
    flow_test = np.array(flow_test)

    # np.random.shuffle(flow_train)

    X_train = flow_train[:, :-1]
    y_train = flow_train[:, -1]
    X_test = flow_test[:, :-1]
    y_test = flow_test[:, -1]

    return X_train, y_train, X_test, y_test


def data_processing(train_path: str, test_path: str, lags: int) -> tuple[np.ndarray,
                                                                         np.ndarray,
                                                                         np.ndarray,
                                                                         np.ndarray,
                                                                         MinMaxScaler]:
    """the main function to process the data

    Args:
        train_path (str): the path of train dataset
        test_path (str): the path of test dataset, actually useless here but necessary
        lags (int): the number of lags which is window size for the flow data.
        That is to say how many days, you use their data to predict the future data

    Returns:
        np.ndarray: the X_train data which is the (lag-1) days each row
        np.ndarray: the y_train data which is the day after (lag-1) days each row
        np.ndarray: the X_test data which is the (lag-1) days each row
        np.ndarray: the y_test data which is the day after (lag-1) days each row
        MinMaxScaler: the scaler used to scale data
    """

    df_train, df_test = _data_reading(train_path, test_path)

    # _data_checking(df_train, df_test)

    flow_train, flow_test, scaler = _data_scaler(df_train, df_test)

    X_train, y_train, X_test, y_test = _traintest_generating(
        flow_train, flow_test, lags)

    return X_train, y_train, X_test, y_test, scaler


# ------------------------------------------------------This is a split line----
if __name__ == '__main__':

    train_path = "data/train.csv"
    test_path = "data/test.csv"

    lags = 12 + 1

    X_train, y_train, X_test, y_test, scaler = data_processing(
        train_path, test_path, lags)
