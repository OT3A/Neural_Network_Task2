import math
from random import shuffle
from numpy import NaN
import numpy as np
import pandas as pd
import csv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pathlib
import cv2
from calendar import month
from re import X
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sympy import true
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score, accuracy_score


def sig(x):
    return 1 / (1 + np.exp(-x))


def initializeWeight(X):
    x = [np.random.uniform(0.00001, 10**(-20)) for i in range(X)]
    # return pd.Series(np.random.rand(X))
    x = pd.Series(x)
    return x

def train(c1, c2, f1, f2, epochs, eta, bias, mse_threshold):
    weights = initializeWeight(3)
    y = c1.iloc[:, 0]
    y = pd.concat([y, c2['species']], axis=0, ignore_index=True)
    y = y.replace([c1.iloc[0,0], c2.iloc[0,0]], [-1,1])
    y = y.astype('int')
    x = c1[[f1, f2]]
    x = pd.concat([x, c2[[f1, f2]]], axis=0, ignore_index=True)
    x0 = np.ones(x.shape[0]) if bias == 1 else np.zeros(x.shape[0])
    x0 = pd.DataFrame(x0, columns = ['bias'])
    x = pd.concat([x0, x], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.40, shuffle=True)
    for i in range(epochs):
        square_error = 0
        train_prediction = []
        for index, row in x_train.iterrows():
            equation = np.dot(weights, row)
            # tempY = np.dot(weights, row)
            # tempY = round(sig(equation))
            tempY = round(equation)
            tempY = -1 if tempY < 0 else 1
            # tempY = -1 if tempY == 0 else tempY
            # if y_train[index] != tempY:
            train_prediction.append(tempY)
            diff = y_train[index] - tempY
            addition = eta * diff * row
            weights = weights.add(addition.tolist())
            # square_error += diff ** 2
        # mse = (1 / x_train.shape[0]) * square_error
        mse = mean_squared_error(y_train, train_prediction)
        # mse = np.square(np.subtract(y_train, train_prediction)).mean()
        train_prediction.clear()
        if mse <= mse_threshold:
            print('mse threshold break')
            break
    prediction = []
    for index, row in x_test.iterrows():
        # prediction.append(round(sig(np.dot(weights, row))))
        prediction.append(round(np.dot(weights, row)))

    # prediction = [-1 if i == 0 else i for i in prediction]
    prediction = [-1 if i < 0 else 1 for i in prediction]
    print(f'prediction = {prediction}\ny test     = {y_test.tolist()}')
    print(f'r2 accuracy = {r2_score(y_test, prediction)}')
    print(f'accuracy = {accuracy_score(y_test, prediction)}')

    # line = []
    # for _, row in x.iterrows():
    #     line.append(np.array((np.dot(weights, row))))
    # # line = [-1 if i == 0 else i for i in line]
    # xLine = [i for i in range(x.shape[0])]
    # plt.scatter(c1[[f1]], c1[[f2]])
    # plt.scatter(c2[[f1]], c2[[f2]])
    # plt.plot(xLine, line)
    # plt.show()

    return weights, accuracy_score(y_test, prediction)

def main(class1, class2, feature1, feature2, epochs=1000, eta=0.1, bias=0, mse=0.5):
    data = pd.read_csv('penguins.csv')
    i = data['gender'].value_counts()

    data['gender'] = data['gender'].replace(['male', 'female', NaN], [0, 1, 0])
    data['gender'] = data['gender'].astype('int')

    # c1 = data.iloc[:50, :]
    # c2 = data.iloc[50:100, :]
    # c3 = data.iloc[100:, :]
    # plt.scatter(c1[[feature1]], c1[[feature2]])
    # plt.scatter(c2[[feature1]], c2[[feature2]])
    # plt.scatter(c3[[feature1]], c3[[feature2]])
    # plt.xlabel(feature1)
    # plt.ylabel(feature2)
    # plt.show()

    # print(c1)
    # print(c2)
    # print(c3)

    # weights = initializeWeight(3)
    # print(f'weights before train = \n{weights}')
    # print(f'weights after train = \n{weights}')
    # print(data.loc[data['species'] == 'Chinstrap'])

    weights, accuracy = train(data.loc[data['species'] == class1], data.loc[data['species'] == class2], feature1, feature2, epochs, eta, bias, mse)
    # print(f'weights = \n{weights}\naccuracy = {accuracy}')
    # with open('accuracy.txt', 'a') as f:
    #     f.write(f'classes ({class1}, {class2}), features ({feature1}, {feature2}), accuracy ({accuracy})\n')
    return weights, accuracy

if __name__ == '__main__':
    w, accuracy = main('Adelie', 'Chinstrap', 'bill_depth_mm', 'bill_length_mm')
    # print(f'accuracy = {accuracy}')
    # print(f'initialized weight = \n{initializeWeight(3)}')