import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

'''
An unsupervised machine learning program that is avaiable through the SciKit learn website
I have gone through the tutorial and tried to get a better understanding of how it works.
'''



# Loading in test data thats avaialble from the sklearn library
digits = load_digits()

# this comes from preprocessing, scale works to standardize a dataset along any axis
data = scale(digits.data)

# gives the ground truth for the digit dataset
y = digits.target


# sets the number of clusters to find, you can do it dynamically or statically like ive done here
# k = len(np.unique(y))
k = 10

# decomposes the data, it pretty much shows us what this data looks like.
samples, features = data.shape


# defining our function to classify our data.
# estimator is the classifier that we will be using, name is just the name and data is what
# we are going to be inputting in, as this is a unsupervised learning we dont have to give it test data

def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
        % (name, estimator.inertia_,
           metrics.homogeneity_score(y, estimator.labels_),
           metrics.completeness_score(y, estimator.labels_),
           metrics.v_measure_score(y, estimator.labels_),
           metrics.adjusted_rand_score(y, estimator.labels_),
           metrics.adjusted_mutual_info_score(y, estimator.labels_),
           metrics.silhouette_score(data, estimator.labels_, metric ='euclidean')))

clf = KMeans(n_clusters=k ,init='random', n_init=10)

bench_k_means(clf, '1', data)