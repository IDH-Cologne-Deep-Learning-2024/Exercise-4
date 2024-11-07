#!/usr/bin/env python3

"""
Implement the logistic function and binary cross-entropy loss function, using titanic data.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def logistic_function(x, a=0, b=0):
    return 1.0 / (1 + np.exp(-(a*x + b)))


# binary cross-entropy loss
def bcel(y, y_hat):
    return -((1/len(y)) * (np.sum((y * np.log(y_hat)) + ((1-y)*(np.log(1-y_hat))))))


def read_data(path):
    dataframe = pd.read_csv(path, sep=",")
    return dataframe


def plot(dataframe, column="Age", a=0, b=0):
    sns.scatterplot(data=dataframe,
                    x=column,
                    y="Survived")
    sns.lineplot(data=dataframe,
                 x=column,
                 y=logistic_function(dataframe.loc[:, column], a, b),
                 label=f"a={a}, b={b}",
                 color="red")
    plt.legend()
    plt.show()


def plot_loss(dataframe, all_a, column="Age", b=0):
    losses = [loss(dataframe, column=column, a=a, b=b) for a in all_a]
    g = sns.lineplot(x=all_a,
                     y=losses,
                     label=f"b={b}")
    g.set(xlabel="parameter a", ylabel="binary cross-entropy loss")
    plt.legend()
    plt.show()


def loss(dataframe, column="Age", a=0, b=0):
    y_hat = logistic_function(dataframe.loc[:, column], a, b)
    return bcel(dataframe.loc[:, "Survived"], y_hat)


def main():
    dataframe = read_data("titanic.csv")

    dataframe = dataframe[dataframe['Age'].notna()]
    dataframe['Age'] = dataframe['Age'] / 10

    print(loss(dataframe, column="Age", a=0.4, b=1))
    plot(dataframe, column="Age", a=0.4, b=1)
    print(loss(dataframe, column="Age", a=0.3, b=-1))
    plot(dataframe, column="Age", a=0.3, b=-1)
    print(loss(dataframe, column="Age", a=1.5, b=-3.5))
    plot(dataframe, column="Age", a=1.5, b=-3.5)
    print(loss(dataframe, column="Age", a=-0.1, b=-0.05))
    plot(dataframe, column="Age", a=-0.1, b=-0.05)

    print(loss(dataframe, column="Pclass", a=0.1, b=1))
    plot(dataframe, column="Pclass", a=0.1, b=1)
    print(loss(dataframe, column="Pclass", a=-0.912, b=1.621))
    plot(dataframe, column="Pclass", a=-0.912, b=1.621)


if __name__ == "__main__":
    main()
