import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


def teeee(num_feats):
    data = pd.read_csv('predict_house.csv')
    X = data.iloc[:, 0:79]
    y = data.iloc[:, -1]
    embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=num_feats)
    embeded_rf_selector.fit(X, y)
    embeded_rf_support = embeded_rf_selector.get_support()
    embeded_rf_feature = X.loc[:, embeded_rf_support].columns.tolist()
    data_train = pd.DataFrame()
    for i in embeded_rf_feature:
        data_train[i] = X[i]
    data_train.to_csv('Selection.csv', index=None)
