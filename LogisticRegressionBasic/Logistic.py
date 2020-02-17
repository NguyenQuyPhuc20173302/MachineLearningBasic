import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

import numpy as np
data = pd.read_csv('data_classification.csv', header=None)
label = data[2]
feature = data.iloc[:, 0:2]
temp = []
for i in range(100):
    temp.append(1)
colum1 = pd.Series(temp)
feature['2'] = colum1


# định nghĩa hàm sigmiod
def sigmoid(z):
    return 1.0/(1+np.exp(-z))

def phanchia(p):
    x = []
    for i in p:
        if(i >= 0.5):
            x.append(1)
        else:
            x.append(0)

    return x

def predict(feature, weight):
    z = np.dot(feature, weight)
    return sigmoid(z)

def cost_function(feature, labels, weight):
    """
    :param feature: 100x3
    :param weight: 3x1
    :param labels: 100x1
    :return:
    """
    n = len(labels)
    prediction = predict(feature, weight)

    cost_1 = -labels*np.log(prediction)
    cost_2 = -(1-labels)*np.log(prediction)

    cost = cost_1+ cost_2
    return cost.sum()/n

def updateWeight(feature, labels, weight, learning_rate):
    """
    :param feature:100x3
    :param labels: 100x1
    :param weight: 3x1
    :param learning_rate: float
    :return:
    """
    n = len(labels)
    prediction = predict(feature, weight)
    gd= np.dot(feature.T, (prediction-labels))
    gd = gd/n
    gd = gd*learning_rate
    weight = weight-gd
    return weight

def train(feature, labels, weight, learning_rate, iter):
    cost_history = []
    for i in range(iter):
        weight = updateWeight(feature, labels, weight, learning_rate)
        cost = cost_function(feature, labels, weight)
        cost_history.append(cost)
    return weight, cost_history

weight = np.array([3., 2., 1.])
learning_rate = 0.01
x, y = train(feature, label, weight, learning_rate, 30000)
kq = predict(feature, x)
kq = phanchia(kq)

from sklearn.metrics import accuracy_score
print("hiệu suât là %.2f %%"%(100*accuracy_score(label, kq)))