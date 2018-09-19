import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
f = open ( 'corporate.txt')
st = f.readline()
pattern = re.compile("^(0|1),([0-9]+),(.*)")
allw = []
labels = []
IND = np.array ([])
L = np.array ([])
while not( st is ''):
    r = pattern.match( st )
    #print (( r.group(1) , r.group(2)))
    if r:
        allw.append(r. group (3))
        IND = np.append ( IND , float ( r.group (2)))
        L = np.append (L , float (r .group (1)))
    st = f.readline ()
cv = CountVectorizer ( max_features =5000 , stop_words = 'english')
I = L == 0
J = [not x for x in I]
A = cv .fit_transform( allw )
feat_names =  cv.get_feature_names ()
u ,s , v = np.linalg.svd(A.toarray () , full_matrices = False )
a = np.dot(u , np.dot(np.diag(s), v ))
#ss = StandardScaler().fit_transform(a)
pca = PCA(n_components=2)
plottabelData = pca.fit_transform(a)
plt.subplot(3,1,1)
plt.title('PCA')
plt.plot(plottabelData[I, 0], plottabelData[I, 1] , '.', color='g')
plt.plot(plottabelData[J, 0], plottabelData[J, 1] , '.', color='r')
plt.subplot(3,1,2)
plt.title('TSNE')
plottabelData = TSNE(n_components=2).fit_transform(a)
plt.plot(plottabelData[I, 0], plottabelData[I, 1] , '.', color='g')
plt.plot(plottabelData[J, 0], plottabelData[J, 1] , '.', color='r')
plt.subplot(3,1,3)
plt.title('KMeans')
plottabelData = KMeans(n_clusters=2).fit_transform(a)
plt.plot(plottabelData[I, 0], plottabelData[I, 1] , '.', color='g')
plt.plot(plottabelData[J, 0], plottabelData[J, 1] , '.', color='r')

plt.show()

# plt.imshow(a)
# plt.show()
