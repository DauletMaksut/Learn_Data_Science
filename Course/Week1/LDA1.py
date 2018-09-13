import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from matplotlib import style
style.use("ggplot")
A = pd.read_csv('bin-classifier-2.txt', header=None)

X = A.values[:, 0:2]
y = A.values[:, 2]

I = y == 1
J = [not x for x in I]

clf = svm.SVC(kernel='linear', C = 1.0 )
clf.fit(X,y)
predict = clf.predict(X)

w = clf.coef_[0]
print(w)
a = -w[0] / w[1]
xx = np.linspace(-3, 3  )
yy = a * xx - clf.intercept_[0] / w[1]

h0 = plt.plot(xx, yy ,'k-')
plt.scatter(X[:, 0], X[:, 1], c = y)
plt.legend()
plt.show()


#print(A)
