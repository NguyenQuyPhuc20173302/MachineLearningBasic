import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
data = pd.read_csv("train.csv")
X = data.iloc[:, 0:20]
y = data.iloc[:, -1]
# apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X, y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
# concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
x = []
for k in range(19):
    best_feature = featureScores.nlargest(k+1, 'Score')
    name_best_feature = best_feature.iloc[:, 0:1]
    best_feature = (name_best_feature["Specs"])
    data_train = pd.DataFrame()
    for i in best_feature:
        data_train[i] = X[i]
    x_train, x_test, y_train, y_test = train_test_split(data_train, y, test_size=0.3)
    rd = RandomForestRegressor(n_estimators=100)
    rd.fit(x_train, y_train)
    result = rd.predict(x_test)
    j = 0
    sum = 0.0

    for i in y_test:
        sum = sum + abs(i - result[j])
        j = j + 1
    x.append(sum/len(result))
plt.plot(x)
plt.show()
