import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# make sure we have tensorflow successuflly installed and running on our device
print(tf.__version__)

# putting the dataset into a dataframe
data = pd.read_csv("student-mat.csv", sep=';')
print('Full dataset shape: ', data.shape)

# arragning the dataframe into the shape we want it and along
# with the correct variables we want to analyze
data = data[['G1', 'G2', 'G3', 'studytime','failures', 'absences']]
print('Data shape: ', data.shape)

# Also known as a Label what we want to predict
predict = 'G3'


# X here is the data without the variable we are tyring to predict
# we are going to try and predict the vale in 'predict', so we need
# to remove it from our test data set
X = np.array(data.drop([predict], 1))

# here is the y value or the actual answer that we will compare our
# guesses of linear regression to.
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

'''

best = 0
for _ in range(30):

    # using SKLEARN's modeling functions to get the values for training and testing the data
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)


    # the actual 'line' that we are trying to predict. I say 'line' since this isn't
    # a line its a value in 5D space. Using Linear Regression
    linear = linear_model.LinearRegression()

    # Fit the data to find the line of best fit using the parametrized data. we can use this
    # to test the data
    linear.fit(x_train, y_train)


    # Now we can use this to see how accurate we are with the data. it will give us a score
    acuracy = linear.score(x_test, y_test)
    print(acuracy)


    if acuracy > best:
        best = acuracy
        # saving the model so we dont have to train it again each time
        with open('studentmodel.pickle', 'wb') as f:
            pickle.dump(linear, f)

'''


# opening our saved file in read mode
pickle_in = open('studentmodel.pickle','rb')

# loading our model in the variable called pickle
linear = pickle.load(pickle_in)


# the coefficents tell us which variables are going to hold the most weight in deciding the score
print('Coeffeceints: ', linear.coef_)

# intercept is going to act the same was an y intercept in a y = mx+b equation === 'b'
print('Intercept: ', linear.intercept_)


predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print('Test values: ', x_test[x],'predicted value = ',round(predictions[x]),'Actual = ', y_test[x])


P = 'studytime'
style.use('ggplot')
pyplot.scatter(data[P], data[predict])
pyplot.xlabel(P)
pyplot.ylabel('Final Grade')
pyplot.show()
