from sklearn import datasets
# take data from sklearn library
wine = datasets.load_wine()

# take data to train and test
from sklearn.model_selection import train_test_split
for i in range(10):
    X_train, X_test, Y_train, Y_test = train_test_split(wine.data, wine.target, test_size=0.3)

# use naive bayer
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score

    gnb = GaussianNB()
    gnb.fit(X_train, Y_train)
    y_pred = gnb.predict(X_test)
    print(accuracy_score(Y_test, y_pred)*100)
