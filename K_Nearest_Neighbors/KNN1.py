import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets

irist = datasets.load_iris()

feature = irist.data
label = irist.target

from sklearn.model_selection import train_test_split


from sklearn.metrics import accuracy_score
for i in range(10):
    x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.25)
    knn = neighbors.KNeighborsClassifier(n_neighbors=9, p=2)
    knn.fit(x_train, y_train)
    y_pre = knn.predict(x_test)
    print("hiệu suất là: %.2f%%"%(100*accuracy_score(y_test, y_pre)))
