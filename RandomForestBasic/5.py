import pandas as pd
from encode import encodeBasic
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
data_set = pd.read_csv('train1.csv',header=None)
data_test = pd.read_csv('test1.csv',header=None)
data_set = data_set.drop(80, axis=1)
data_set = data_set.append(data_test)
data_set = data_set.iloc[:, 1:80]
test = data_set[1460:]

