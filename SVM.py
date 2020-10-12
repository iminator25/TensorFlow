import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


''''
A working example of using support-vector machines (SVM), a supervised machine learning method
one can use to predict values. I am using an example dataset that is provided by sklearn
'''

# loading the dataset in
cancer = datasets.load_breast_cancer()

# print(cancer.feature_names)
# print(cancer.target_names)

# Getting the label and the features to use to learn from
X = cancer.data
y = cancer.target

# splitting up the data into the form that sklearn likes to so we can easily train it
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

# print(x_train, y_train)
# classes = ['malignant' 'benign']


# creating the classifier:
# here we can choose the kernel function to be linear polynomial, rbf etc.
# the higher the degree the longer it will take. the 'C'value is the
# regularization parameter. Must be positive basically  to prevent overfitting.
# its default is 1.
clf = svm.SVC(kernel = 'poly',degree=2, C=2)

# fits the model given the training data
clf.fit(x_train, y_train)

# preforms the classification. pretty much gives the predicted value
y_pred = clf.predict(x_test)

# uses sklearns handy function to test how accurate the predcition is.
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)