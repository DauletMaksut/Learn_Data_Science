

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

A = pd.read_csv("data_lda_circular.txt", header=None)

X = A.values[:, 0:2]
y = A.values[:, 2]

# indices for each class
ind1 = (y == 1)
ind2 = (y == 2)

#plt.subplot(5,2, sharex='col', sharey='row')
for z in range(1, 16):
    clf = DecisionTreeClassifier(max_depth = z)
    clf.fit(X,y)
    scores = cross_val_score(clf, X, y, cv=5)
    print("The average score is {x}.".format(x=np.mean(scores)))
    err = clf.predict(X)
    print(np.sum(err))
    err = (err != y)
    plt.subplot(5,3 ,z)
    name = 'Max depth:' + str(z)
    plt.title(name)
    plt.plot(X[ind1,0],X[ind1,1],'.')
    plt.plot(X[ind2,0],X[ind2,1],'.')
    plt.plot(X[err ,0],X[err,1],'*', color='r')
plt.grid()
plt.show()
