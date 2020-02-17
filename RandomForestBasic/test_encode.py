import pandas as pd
from encode import encodeBasic
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing

data_set = pd.read_csv('train1.csv', header=None)
label = data_set[80]
data_test = pd.read_csv('test1.csv')
data_set = data_set.drop(80, axis=1)
data_set = data_set.append(data_test)
feature = data_set.iloc[:, 1:80]

feature = feature[1:]

label = label[1:1460]
print(feature)
for i in range(79):
    for j in feature[i+1]:
        try:
            tem = int(j)
        except:
            feature[i+1] = encodeBasic.encode(feature[i+1])
            break


'''
for i in range(79):
    for j in feature[i + 1]:
        try:
            tem = int(j)
        except:
            feature[i + 1] = encodeBasic.encode(feature[i + 1])
            break
'''

# feature_train = feature[1:1460]
# feature_test = feature[1460:]

for i in range(1459):
    label[i + 1] = int(label[i + 1])

'''
x_train, x_test, y_train, y_test = train_test_split(feature_train, label)
re = RandomForestRegressor(n_estimators=100)
#re = GradientBoostingRegressor()
re.fit(x_train, y_train)
y_pre = re.predict(feature_test)
print(y_pre)
'''
