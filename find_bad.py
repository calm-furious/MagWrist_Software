# %%
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import decomposition
from sklearn import datasets
from cjc_util import *
# %%
b_multi = []
b_null_multi = []
for trial in range(10):
    b_multi.append(np.load('../data/cjc1_%d_multiple.npy' % trial))
    b_null_multi.append(np.load('../data/cjc1_%d_null_multiple.npy' % trial))

# %%\


finger_code_list = [24, 20, 18, 12, 10, 6]
Y = np.arange(10).repeat(20)
Y = np.concatenate((Y, np.arange(10, 20).repeat(20)))

index = 10

X = np.empty([0, 30])
for trial in range(10):
    X = np.vstack((
        X,
        b_multi[trial][index] -
        np.mean(b_null_multi[trial][index*4:index*4+4], axis=0)
    ))

index = 24
for trial in range(10):
    X = np.vstack((
        X,
        b_multi[trial][index] -
        np.mean(b_null_multi[trial][index*4:index*4+4], axis=0)
    ))

print(X.shape)
print(Y)
pca = decomposition.PCA(n_components=2, svd_solver='full')
pca.fit(X)
projected = pca.transform(X)

# 画出每个点的前两个主成份
plt.scatter(projected[:, 0], projected[:, 1], c=Y+1, edgecolor='none',

            alpha=1, cmap=plt.cm.get_cmap('Spectral', 20))
# plt.xlim((-1000, 1000))
# plt.ylim(ymax=400)
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()


# %%
