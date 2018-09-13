import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score



tt = pd.read_csv('bin-classifier-2.txt', header=None)
X = tt.values[:, 0:2]
y = tt.values[:, 2]
I = y == 1
# list comprehension
J = [not x for x in I]
clf = LinearDiscriminantAnalysis()
clf.fit(X,y)
err = clf.predict(X) != y
# computing the mean vectors
def line(clf):
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-2 , 3  )
    yy = a * xx - clf.intercept_[0] / w[1]
    return xx, yy
xx, yy = line(clf)
comparison = np.vstack((clf.predict(X), y)).T
plt.title('LDA with line')
plt.plot(xx, yy, 'k-')
plt.plot(X[I,0],X[I,1],'.', color='g')
plt.plot(X[J,0],X[J,1],'.', color='b')
plt.plot(X[err,0],X[err,1],'.', color = 'r')
plt.grid()
plt.show()

#Accuracy
print("Accuracy: ",accuracy_score(clf.predict(X), y))
#precision
print("Precision: ",precision_score(clf.predict(X), y))
#recall
print("Recall: ",recall_score(clf.predict(X), y))
