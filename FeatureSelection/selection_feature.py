import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def selection_f(data, n):
    chieudai = len(data.columns)
    X = data.iloc[:, 0:chieudai-1]
    y = data.iloc[:, -1]
    bestfeatures = SelectKBest(score_func=chi2, k=n)
    fit = bestfeatures.fit(X, y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']
    best_feature = featureScores.nlargest(n, 'Score')
    name_best_feature = best_feature.iloc[:, 0:1]
    best_feature = (name_best_feature["Specs"])
    data_train = pd.DataFrame()
    for i in best_feature:
        data_train[i] = X[i]
    return data_train
