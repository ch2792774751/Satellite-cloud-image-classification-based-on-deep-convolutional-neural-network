import scipy.io
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import sqrt


def save_data(file_dir, filename, data):
    #scipy.io.savemat(file_dir + filename + '.mat', {filename: data})
    save = pd.DataFrame(data, columns=[filename])
    save.to_csv(file_dir + filename + '.csv', index=False)

def plot_fig(data1, data2, label1, label2):
    plt.plot(data1, label=label1)
    plt.plot(data2, label=label2)
    plt.legend()
    plt.show()

def mean_squared_error(y_true, y_predict):
    """计算y_true和y_predict之间的MSE"""
    assert len(y_true) == len(y_predict), \
        "the size of y_true must be equal to the size of y_predict"
    return np.sum((y_true - y_predict)**2) / len(y_true)


def root_mean_squared_error(y_true, y_predict):
    """计算y_true和y_predict之间的RMSE"""
    return sqrt(mean_squared_error(y_true, y_predict))


def mean_absolute_error(y_true, y_predict):
    """计算y_true和y_predict之间的MAE"""
    assert len(y_true) == len(y_predict), \
        "the size of y_true must be equal to the size of y_predict"
    return np.sum(np.absolute(y_true - y_predict)) / len(y_true)


def r2_score(y_true, y_predict):
    """计算y_true和y_predict之间的R Square"""
    return 1 - mean_squared_error(y_true, y_predict) / np.var(y_true)
