import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("predict_house.csv")
X = data.iloc[:, 0:79]  # independent columns
y = data.iloc[:, -1]  # target column i.e price range
# get correlations of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20, 20))
# plot heat map
g = sns.heatmap(X[top_corr_features].corr(), annot=True, cmap="RdYlGn")
plt.show()
