import numpy as np
import pandas as pd
import math
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import linear_model


tt = pd.read_csv("data.txt" , header = None )
x= tt.values[: ,0]
y= tt.values[: ,1]
x = x[:, np.newaxis]
# X = np.hstack((0* x + 1, x ))

x = x * 0 + 1
N = len(x)
aics = [0 for x in range(1,12)] 
for p in range(1, 12):
    x = np.hstack((x, x ** p))
    clr = linear_model.LinearRegression ()
    clr.fit(x , y)
    y_pred = clr.predict (x)
    E = np.sum((y - y_pred) ** 2)
    aic = N * math.log(E / N) + 2 * (p + 2)
    aics[p-1] = aic
p_min = np.argmin(aics)
print("p_min=%d aic=%d" %(p_min + 1 , aics[p_min]))
plt.plot(aics)
plt.show()

