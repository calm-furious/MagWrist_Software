# %%
# headers

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
# data loader
train_num = 5
test_num = 5
total_num = 10
# data list
b = []

for index in range(0, total_num):
    b.append(np.load('../data/cjc_%d_single.npy' % index))


# %%
# axes_array = []
# subplot_row = 4
# subplot_col = 3

# axes_array.append(fig.add_subplot(subplot_row, subplot_col, 3))
# axes_array.append(fig.add_subplot(subplot_row, subplot_col, 2))
# axes_array.append(fig.add_subplot(subplot_row, subplot_col, 1))
# axes_array.append(fig.add_subplot(subplot_row, subplot_col, 6))
# axes_array.append(fig.add_subplot(subplot_row, subplot_col, 4))
# axes_array.append(fig.add_subplot(subplot_row, subplot_col, 9))
# axes_array.append(fig.add_subplot(subplot_row, subplot_col, 7))
# axes_array.append(fig.add_subplot(subplot_row, subplot_col, 12))
# axes_array.append(fig.add_subplot(subplot_row, subplot_col, 11))
# axes_array.append(fig.add_subplot(subplot_row, subplot_col, 10))
# axes_array.append(fig.add_subplot(subplot_row, subplot_col, 8))

# %%
# pca
X = np.empty([0, 30])
trial_list = [2, 3, 4, 5, 7, 8]
for trial in trial_list:
    for i in range(5):
        idx = i + 1
        X = np.vstack((X,
                       b[trial][idx] -
                       np.mean(b[trial][0][(idx-1)*4:idx*4], axis=0)
                       ))
Y = np.array([0, 1, 2, 3, 4])
Y = Y.repeat(20)
Y = np.tile(Y, len(trial_list))
print(X.shape, Y.shape)

pca = decomposition.PCA(n_components=2, svd_solver='full')
projected = pca.fit_transform(X)

# 画出每个点的前两个主成份
plt.scatter(projected[:, 0], projected[:, 1], c=Y+1, edgecolor='none',
            alpha=1, cmap=plt.cm.get_cmap('Spectral', 5))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()


# %%
# 极差
for trial in range(0, 10):
    for i in range(5):
        idx = i + 1
        temp = np.ptp(b[trial][idx] - np.mean(b[trial][0]
                                              [(idx - 1) * 4:idx * 4], axis=0), axis=0)
        flag = np.abs(temp) > 1000
        if (True in flag):
            print("trial: %d, finger id:%d   " % (trial, i), temp)

# %%
trial = 1
idx = 4
temp = b[trial][idx] - np.mean(b[trial][0][(idx-1)*4:idx*4], axis=0)
temp[:, 19]
# for trial in range(5, 10):
#     for i in range(5):
#         idx = i + 1
#         temp = b[trial][idx] - np.mean(b[trial][0][(idx-1)*4:idx*4], axis=0)
#         temp[]


# %%
X_multi = np.empty([0, 30])  # [ [],[],...,[] ]
for trial in [2, 3, 4, 5, 7, 8]:  # [3, 5, 6, 7, 8, 9]:
    for i in [1, 2, 3]:
        for j in [2, 3, 5]:
            if (j <= i):
                continue
            # combine into new feature
            # b[trial][i] (20*30) + b[trail][j] (20*30)
            X_multi = np.vstack((X_multi, b[trial][i] + b[trial][j] -
                                 np.mean(b[trial][0][(i-1)*4:i*4], axis=0) -
                                 np.mean(b[trial][0][(j - 1) * 4:j * 4], axis=0)))
Y_multi = np.array([8, 9, 5, 10, 6, 7])
Y_multi = Y_multi.repeat(20)
Y_multi = np.tile(Y_multi, 6)
Y_multi.shape

pca = decomposition.PCA(n_components=2, svd_solver='full')
projected = pca.fit_transform(X_multi)

# 画出每个点的前两个主成份
plt.scatter(projected[:, 0], projected[:, 1], c=Y-4, edgecolor='none',
            alpha=1, cmap=plt.cm.get_cmap('Spectral', 6))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()


# %%
