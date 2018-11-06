# import statements , loading files , and declarations
import pandas as pd
import numpy as np
from sklearn import linear_model


BTRAIN = pd.read_csv("train.csv", header=None).values
XTest = pd.read_csv("test.csv", header=None).values
X = BTRAIN[:, 1:101]
y = BTRAIN[:, 101:121]
XTest = BTESTDIST[m, 1:]
XTest = XTest[np.newaxis, :]
clr = linear_model.LinearRegression()
clr.fit(X, y)
yp = clr.predict(XTest)