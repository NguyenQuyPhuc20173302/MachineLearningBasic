from __future__ import print_function
import numpy as np
from sklearn import datasets, linear_model

from genetic_selection import GeneticSelectionCV
import pandas as pd

def main():
    data = pd.read_csv('predict_house.csv')
    # Some noisy data not correlated
    E = np.random.uniform(0, 0.1, size=(len(data), 20))

    X = data.iloc[:, 0:79]
    y = data.iloc[:, -1]

    estimator = linear_model.LogisticRegression(solver="liblinear", multi_class="ovr")

    selector = GeneticSelectionCV(estimator,
                                  cv=5,
                                  verbose=1,
                                  scoring="accuracy",
                                  max_features=5,
                                  n_population=50,
                                  crossover_proba=0.5,
                                  mutation_proba=0.2,
                                  n_generations=40,
                                  crossover_independent_proba=0.5,
                                  mutation_independent_proba=0.05,
                                  tournament_size=3,
                                  n_gen_no_change=10,
                                  caching=True,
                                  n_jobs=-1)
    selector = selector.fit(X, y)
    kq = selector.predict(X)
    j = 0
    for i in y:
        print(i)
        print(kq[j])
        print()
        j = j + 1

if __name__ == "__main__":
    main()