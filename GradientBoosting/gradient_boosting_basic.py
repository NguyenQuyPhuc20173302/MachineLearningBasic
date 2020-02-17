import pandas as pd
from encode import encodeBasic
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing
data_set = pd.read_csv('train1.csv', header=None)

feature = data_set.iloc[:, 1:80]
feature = feature[1:1460]
label = data_set[80]
label = label[1:1460]
x = 'jhashfs'
for i in range(79):
    for j in feature[i+1]:
        try:
            tem = int(j)
        except:
            feature[i+1] = encodeBasic.encode(feature[i+1])
            break

for i in range(1459):
   label[i+1] = int(label[i+1])

for i in range(10):
    x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.3)
    #re = RandomForestRegressor(n_estimators=1000)
    re = GradientBoostingRegressor()
    re.fit(x_train, y_train)
    y_pre = re.predict(x_test)
    j = 0
    sum = 0.0
    for i in y_test:
        sum = sum + abs(i - y_pre[j])/i
        j = j + 1
    print(sum/len(y_pre))
