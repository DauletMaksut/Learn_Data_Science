import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

corpus = [' To be , or not to be , that is the question ',
' Whether tis nobler in the mind to suffer ',
' The slings and arrows of outrageous fortune ',
'Or to take arms against a sea of troubles ',
' And by doing something ',
' the the the the the the the '
]
vectorizer = CountVectorizer ( min_df =1)
dt = vectorizer.fit_transform ( corpus )
feat_names = vectorizer.get_feature_names ();
u ,s , v = np.linalg.svd(dt.toarray () , full_matrices = False )
a = np.dot(u , np.dot(np.diag(s), v ))
up = len(s)
s_temp = [0] * up
for x in range(1, up + 1):
    s_temp[x-1] = s[x-1]
    a = np.dot(u , np.dot(np.diag(s_temp), v ))
    plt.subplot(6,1,x)
    plt.title('With ' + str(x) +' elements')
    plt.imshow(a)
plt.show()
