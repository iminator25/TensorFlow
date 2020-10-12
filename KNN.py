import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing


# K Nearest neighbors implementation. A simple, supervised machine learning
# tool to predict values. Run as many times as you want to increase accuracy.

# reading in the data from downloaded csv provided by UCI
data = pd.read_csv('car.data')
print(data.head())

# using a tool from sklearn that allows us to manipulate the header of the CSV
# to be in a way that we can use it. since all the values are not nice numbers.
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data['buying']))
maint = le.fit_transform(list(data['maint']))
door = le.fit_transform(list(data['door']))
persons = le.fit_transform(list(data['persons']))
lug_boot = le.fit_transform(list(data['lug_boot']))
safety = le.fit_transform(list(data['safety']))
cls = le.fit_transform(list(data['class']))

print(buying)

# choosing the value that we would like to predict given other values
predict = 'class'

# An easy way to put the values in the correct shape so that the
# sklearn model selection tool can accept them.
X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

# splitting up the data into testing and training so that sklearn can easily use it.
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# choosing the amount of nieghboors we want to use, its different for each dataset and requires
# prior knowledge of the data to make an informed decision. generally >=9 is too many.
model = KNeighborsClassifier(n_neighbors=9)

# fitting the model with the test data.
model.fit(x_train, y_train)

# scoring the model, testing vs training model
accuracy = model.score(x_test, y_test)
print(accuracy)

# creating our predicition from the test data
predicted = model.predict(x_test)

# an array that will more give us names just like the original data has.
names = ['unacc', 'acc', 'good', 'vgood']

# Revewing each of the scores of the data to see which points worked as expected
# and which ones did not.
for x in range(len(x_test)):
    print('Predicted: ', names[predicted[x]], 'Data: ', x_test[x], 'Actual: ', names[y_test[x]] )
    n = model.kneighbors([x_test[x]], 9, True)
    print('N: ', n)