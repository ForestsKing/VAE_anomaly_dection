import warnings

import numpy as np

warnings.filterwarnings("ignore")


def getdata(df):
    normal = df[df['anomaly'] == 0]
    anomal = df[df['anomaly'] == 1]

    train_x = normal.drop(['anomaly', 'changepoint'], axis=1).values[:400]
    test_x = np.vstack(
        (normal.drop(['anomaly', 'changepoint'], axis=1).values[400:], anomal.drop(['anomaly', 'changepoint'], axis=1)))
    train_y = np.zeros(len(train_x))
    test_y = np.hstack((np.zeros(len(test_x) - len(anomal)), np.ones(len(anomal))))

    shuffle_ix = np.random.permutation(np.arange(len(train_x)))
    train_x = train_x[shuffle_ix]
    train_y = train_y[shuffle_ix]

    shuffle_ix = np.random.permutation(np.arange(len(test_x)))
    test_x = test_x[shuffle_ix]
    test_y = test_y[shuffle_ix]

    return train_x.reshape(len(train_x), -1), test_x.reshape(len(test_x), -1), train_y, test_y
