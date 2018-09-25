import numpy as np
import pandas as pd

tmp = pd.read_csv('bb-data.txt', header=None)
xi = tmp.values[:,0:6]
ti = tmp.values[:,6:12]
A = np.identity(6)

for x in range(len(tmp.values[:,1])):
    err = ti[x] - np.dot(A, xi[x])
    A += np.dot( (ti[x] - err), (xi[x]).T)
print(A)
print(err)
