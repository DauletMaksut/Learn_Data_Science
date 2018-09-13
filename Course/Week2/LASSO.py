import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd

A = pd.read_csv("lasso_example.txt",header=None)

x=A.values[:,1]
X=A.values[:,0:4]
y=A.values[:,4]

valueOfAlpha = 10
while valueOfAlpha >= 1e-5:
	lasmodel = linear_model.Lasso(alpha=valueOfAlpha,fit_intercept=False)
	lasmodel.fit(X,y)
	print( lasmodel.coef_)
	value = lasmodel.coef_
	plt.plot(valueOfAlpha, value[0], '.', color = 'grey')
	plt.plot(valueOfAlpha, value[1], '.', color = 'blue')
	plt.plot(valueOfAlpha, value[2], '.', color = 'red')
	plt.plot(valueOfAlpha, value[3], '.', color = 'green')
	valueOfAlpha /= 1.01

print(dir(plt))
plt.grid()
plt.show()
print(lasmodel.coef_)

###
