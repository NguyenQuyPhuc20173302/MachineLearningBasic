import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
data_set = pd.read_csv('data_classification.csv',header=None)
feature = data_set.iloc[:, 0:2]
label = data_set[2]
sum = 0.0
for i in range(100):
    x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.3)

    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_pre = gnb.predict(x_test)
    sum = sum + accuracy_score(y_test, y_pre)

print(sum)