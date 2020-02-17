import numpy as np
import pandas as pd

data_set = pd.read_csv('data_classification.csv', header=None)
feature = data_set.iloc[:, 0:2]
label = data_set[2]

from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.metrics import accuracy_score
sum = 0.0
for i in range(100):
    x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.3)
    knn = neighbors.KNeighborsClassifier(n_neighbors=5, p=2)
    knn.fit(x_train, y_train)
    y_pre = knn.predict(x_test)
    sum = sum + accuracy_score(y_test, y_pre)

print(sum)