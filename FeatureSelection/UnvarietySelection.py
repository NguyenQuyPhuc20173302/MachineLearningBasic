from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, univariate_selection
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from FeatureSelection import performance
data = pd.read_csv('predict_house.csv')
X = data.iloc[:, 0:79]
Y = data.iloc[:, -1]
for i in range(75):
    print(i+1)
    test = SelectKBest(score_func=f_classif, k=i+1)
    fit = test.fit(X, Y)
    # summarize scores
    set_printoptions(precision=3)
    features = fit.transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    tpot = GradientBoostingRegressor()
    tpot.fit(x_train, y_train)
    result = tpot.predict(x_test)
    print(performance.performance(y_test, result))

