#!/usr/bin/env python3

"""
Implement the logistic function and binary cross-entropy loss function, using titanic data.
"""

import numpy as np
import pandas as pd
import seaborn as sns



def logistic_function(x, a=0, b=0):
    pass


# binary cross-entropy loss
def bcel(y, y_hat):
    y_hat = np.clip(y_hat, 1e-7, 1 - 1e-7)
    term_0 = (1 - y) * np.log(1 - y_hat + 1e-7)
    term_1 = y * np.log(y_hat + 1e-7)
    return -np.mean(term_0 + term_1, axis=0)


def read_data(file="titanic.csv"):
    dataframe = pd.read_csv(file, header=0)
    print(dataframe)


def plot(dataframe):
    sns.catplot(data=dataframe, kind="bar", x="Age", y="Survived") #TODO: change bar
    sns.catplot(data=dataframe, kind="bar", x="Pclass", y="Survived")


def loss(dataframe):
    pass
