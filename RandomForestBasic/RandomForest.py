import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
data_set = pd.read_csv('X_train.csv', header=None)
feature = data_set.iloc[:, 0:79]
label = data_set[79]


sum = 0.0
for i in range(10):
    x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.3)
    re = RandomForestRegressor(n_estimators=500)
    re.fit(x_train, y_train)
    y_pre = re.predict(x_test)
    j = 0
    sum = 0.0
    for i in y_test:
        sum = sum + 1 - abs(i - y_pre[j]) / i
        j = j + 1
    print(sum / len(y_pre))

