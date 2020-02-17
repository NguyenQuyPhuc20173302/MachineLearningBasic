weather = ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Rainy', 'Sunny',
           'Overcast', 'Overcast', 'Rainy']
temp = ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild']

play = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']

from sklearn import preprocessing
import array as arr

le = preprocessing.LabelEncoder()
temp = le.fit_transform(temp)
weather = le.fit_transform(weather)
label = le.fit_transform(play)
feature = []

for i in range(len(weather)):
    x = [weather[i], temp[i]]
    feature.append(x)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(feature, label)
test = [[0, 2]]
prediction = model.predict(test)
print(prediction)