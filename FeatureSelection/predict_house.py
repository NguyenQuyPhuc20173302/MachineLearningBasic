import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt
from FeatureSelection import selection_feature
from FeatureSelection import performance
data = pd.read_csv("predict_house.csv")
X = data.iloc[:, 0:79]
y = data.iloc[:, -1]
x = []
data_train = selection_feature.selection_f(data, 70)
print(data_train)
x_train, x_test, y_train, y_test = train_test_split(data_train, y, test_size=0.3)
rd = GradientBoostingRegressor()
rd.fit(x_train, y_train)
result = rd.predict(x_test)
print(performance.performance(y_test, result))







