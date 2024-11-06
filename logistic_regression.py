#!/usr/bin/env python3

"""
Implement the logistic function and binary cross-entropy loss function, using titanic data.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def logistic_function(x, a=0, b=0):
    lg = 1 / (1 + np.exp(-(a * x + b)))
    return lg


# binary cross-entropy loss
def bcel(y, y_hat):
    bceloss = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    return bceloss


def read_data(path):
    dataframe = pd.read_csv(path)
    return dataframe


def plot(dataframe):
    plt.subplot(2,1,1)
    sns.scatterplot(data=dataframe, x = "Age", y = "Survived")
    plt.xlabel("Age")
    plt.ylabel("Did they suirvive?")
    plt.title("Age/Survived")
    
    plt.subplot(2,1,2)
    sns.scatterplot(data=dataframe, x="Pclass",y="Survived")
    plt.title("PassengerClass/Survived")
    
    plt.show()


def loss(dataframe):
    pass

file = read_data("titanic.csv")
plot(file)