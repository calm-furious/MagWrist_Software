# %%
# headers

import time
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import decomposition
from sklearn import datasets
from cjc_util import *
import itertools
from sklearn.metrics import accuracy_score
# %%
# ----------------------------- data loader --------------------------------------
train_num = 5
test_num = 5
total_num = 10
# data list
b = []

for index in range(0, total_num):
    b.append(np.load('../data/cjc1_%d_single.npy' % index))
'''
# b1 = np.load('cjc_1_single.npy')
# b2 = np.load('cjc_2_single.npy')
# b3 = np.load('cjc_3_single.npy')
'''

# ---------------------- construct train-test vectors/matrices -----------------------
# using all 20 batches
X = np.empty([0, 30])
# for trial in [2, 3, 4, 5, 7, 8]:
for trial in range(10):
    for i in range(5):
        idx = i + 1
        if(idx == 4):
            continue
        X = np.vstack((X,
                       b[trial][idx] -
                       np.mean(b[trial][0][(idx-1)*4:idx*4], axis=0)
                       ))

Y = np.array([0, 1, 2,  4])
Y = Y.repeat(20)
Y = np.tile(Y, 10)  # 6)
print(Y.shape)

# -------------------------------  train test split -----------------------------
x_train = X[:int(X.shape[0]/2)]
x_test = X[int(X.shape[0]/2):]

y_train = Y[:int(Y.shape[0]/2)]
y_test = Y[int(Y.shape[0] / 2):]

# x_train = X[:int(X.shape[0]*4/5)]
# x_test = X[int(X.shape[0]*4/5):]

# y_train = Y[:int(Y.shape[0]*4/5)]
# y_test = Y[int(Y.shape[0]*4/5):]
'''
X = np.vstack((x_train, x_test))
Y = np.hstack((y_train, y_train))
x_train1, x_test1, y_train1, y_test1 = train_test_split(
    X, Y, random_state=10, train_size=0.5, test_size=0.5)
'''

### Pre - Processing ###
ave_cols = x_train.mean(axis=0)
std_cols = x_train.std(axis=0)
X_norm = (x_train - ave_cols) / std_cols
zero_col_list = np.where(np.isnan(X_norm))[1]
X_norm = np.delete(X_norm, np.where(np.isnan(X_norm))[1], axis=1)
print(X_norm.shape)
x_train = X_norm

# ave_cols = x_test.mean(axis=0)
# std_cols = x_test.std(axis=0)
x_test_norm = (x_test - ave_cols) / std_cols
x_test_norm = np.delete(x_test_norm, np.where(
    np.isnan(x_test_norm))[1], axis=1)
x_test = x_test_norm
# y_train = Y[:int(Y.shape[0]/2)]
# y_test = Y[int(Y.shape[0]/2):]

# PCA
# X = np.vstack((x_train, x_test))
# pca = decomposition.PCA(n_components=0.9999)
# pca.fit(X)
# x_train_pca = pca.transform(x_train)
# x_test_pca = pca.transform(x_test)

# # normalize
# ------------------------------ training & testing --------------------------------


clf = svm.SVC(gamma=0.001, C=10000)
# clf = svm.SVC(decision_function_shape='ovo')

# Ada boost
# clf = AdaBoostClassifier(n_estimators=100, random_state=0)

# # Random forest
# clf = RandomForestClassifier(n_estimators=150, max_depth=30, random_state=2)

# # KNN
# clf = KNeighborsClassifier(n_neighbors=4)


####### training #########
y_train = combine(y_train)
y_test = combine(y_test)

clf.fit(x_train, y_train)
print(clf.score(x_train, y_train), clf.score(x_test, y_test))
v2_names = ['食指', '中指', '无名指', 'little',
            '拇指', '拇指和食指', '拇指和中指', '拇指和无名指', '食指和中指', '食指和无名指', '中指和无名指']
plot_confusion_matrix(y_test, list(clf.predict(x_test)),
                      classes=np.array(v2_names), normalize=True, title='单手指手势混淆矩阵')
# clf.fit(x_train_pca, y_train)
# print(clf.score(x_test_pca, y_test))
# plot_confusion_matrix(y_test, list(clf.predict(x_test_pca)),
#                       classes=v2_names, normalize=True, title='')

# multi-finger
# ----------------------- constructing simulated data ---------------------------------
# two fingers
# 12 13 T1 23 T2 T3
# 8  9  5  10  6  7
# 12 10 24 6   20 18
# 20*30
X_multi_fake = np.empty([0, 30])  # [ [],[],...,[] ]
for trial in range(10):  # [2, 3, 4, 5, 7, 8]:
    for [i, j] in [[5, 1], [5, 2], [5, 3], [1, 2], [1, 3], [2, 3]]:
        # print(i, j)
        # combine into new feature
        # b[trial][i] (20*30) + b[trail][j] (20*30)
        X_multi_fake = np.vstack((X_multi_fake, b[trial][i] + b[trial][j] -
                                  np.mean(b[trial][0][(i-1)*4:i*4], axis=0) -
                                  np.mean(b[trial][0][(j - 1) * 4:j * 4], axis=0)))
Y_multi_fake = np.array([5, 6, 7, 8, 9, 10])
Y_multi_fake = Y_multi_fake.repeat(20)
Y_multi_fake = np.tile(Y_multi_fake, 10)


X_multi_train = X_multi_fake
Y_multi_train = Y_multi_fake

X_norm = (X_multi_train - ave_cols) / std_cols
zero_col_list = np.where(np.isnan(X_norm))[1]
X_norm = np.delete(X_norm, np.where(np.isnan(X_norm))[1], axis=1)
print(X_norm.shape)
X_multi_train = X_norm

X_multi_fake = X_multi_train

# -------------------------- load multi-finger data ----------------------------
finger_code_list = [24, 20, 18, 12, 10, 6]

b_multi = []
b_null_multi = []
for trial in range(10):
    b_multi.append(np.load('../data/cjc1_%d_multiple.npy' % trial))
    b_null_multi.append(np.load('../data/cjc1_%d_null_multiple.npy' % trial))
test = [4, 9, 7, 2]
X_multi_test = np.empty([0, 30])
for index in finger_code_list:
    for trial in test:
        X_multi_test = np.vstack((
            X_multi_test,
            b_multi[trial][index] -
            np.mean(b_null_multi[trial][index*4:index*4+4], axis=0)
        ))

print(X_multi_test.shape)
Y_multi_test = np.array([5, 6, 7, 8, 9, 10])
Y_multi_test = Y_multi_test.repeat(20*len(test))
print(Y_multi_test.shape)
#

X_multi_test_norm = (X_multi_test - ave_cols) / std_cols
X_multi_test_norm = np.delete(X_multi_test_norm, np.where(
    np.isnan(X_multi_test_norm))[1], axis=1)
X_multi_test = X_multi_test_norm

# -------------------------  start of fitting ---------------------------------
# clf2 = RandomForestClassifier(n_estimators=150, max_depth=30, random_state=2)
# clf2 = svm.SVC(gamma=0.9, C=0.1)
clf2 = svm.SVC(gamma=0.001, C=10000)

# clf2 = KNeighborsClassifier(n_neighbors=4, p=3)
clf2.fit(X_multi_train, Y_multi_train)
print(clf2.score(X_multi_test, Y_multi_test))
v2_names = ['食指', '中指', '无名指', 'little',
            '拇指', '拇指和食指', '拇指和中指', '拇指和无名指', '食指和中指', '食指和无名指', '中指和无名指']

v2_names = np.array(v2_names)
plot_confusion_matrix(Y_multi_test, list(clf2.predict(X_multi_test)), _size=13,
                      classes=v2_names, normalize=True, title='单双指全手势混淆矩阵（叠加特征）')
# train-test single & double

# %%
train_ratio = 0.5
test_ratio = 0.5
finger_code_list = [24, 20, 18, 12, 10, 6]
X_multi = np.empty([0, 30])
trail_list = [1, 3, 6, 5, 4, 9, 7, 2]
for trial in trail_list:
    for index in finger_code_list:
        X_multi = np.vstack((
            X_multi,
            b_multi[trial][index] -
            np.mean(b_null_multi[trial][index*4:index*4+4], axis=0)
        ))

Y_multi = np.array([5, 6, 7, 8, 9, 10]).repeat(20)
Y_multi = np.tile(Y_multi, len(trail_list))


X_multi_train = X_multi[:int(X_multi.shape[0]*train_ratio)]
X_multi_test = X_multi[int(X_multi.shape[0]*test_ratio):]

Y_multi_train = Y_multi[:int(Y_multi.shape[0]*train_ratio)]
Y_multi_test = Y_multi[int(Y_multi.shape[0] * test_ratio):]


# ave_cols = X_multi_train.mean(axis=0)
# std_cols = X_multi_train.std(axis=0)
X_norm = (X_multi_train - ave_cols) / std_cols
zero_col_list = np.where(np.isnan(X_norm))[1]
X_norm = np.delete(X_norm, np.where(np.isnan(X_norm))[1], axis=1)
print(X_norm.shape)
X_multi_train = X_norm

X_multi_test_norm = (X_multi_test - ave_cols) / std_cols
X_multi_test_norm = np.delete(X_multi_test_norm, np.where(
    np.isnan(X_multi_test_norm))[1], axis=1)
X_multi_test = X_multi_test_norm

# clf = svm.SVC(kernel='poly', gamma=5, C=10)
# clf = RandomForestClassifier(n_estimators=150, max_depth=30, random_state=2)
clf = KNeighborsClassifier(n_neighbors=10, p=3)
clf.fit(X_multi_train, Y_multi_train)
print(clf.score(X_multi_train, Y_multi_train),
      clf.score(X_multi_test, Y_multi_test))
v2_names = ['食指', '中指', '无名指', 'little',
            '拇指', '拇指和食指', '拇指和中指', '拇指和无名指', '食指和中指', '食指和无名指', '中指和无名指']
v2_names = np.array(v2_names)
plot_confusion_matrix(Y_multi_test, list(clf.predict(X_multi_test)), _size=13,
                      classes=v2_names, normalize=True, title='')

# %%
# ------------------------- comprehensive ---------------------
X_com_train = np.vstack((x_train, X_multi_fake))  # , X_multi_train2))
X_com_test = np.vstack((x_test, X_multi_test))  # , X_multi_test2))
Y_com_train = np.concatenate((y_train, Y_multi_fake))  # , Y_multi_train2))
Y_com_test = np.concatenate((y_test, Y_multi_test))  # , Y_multi_test2))

# Y_com_train = combine(Y_com_train)
# Y_com_test = combine(Y_com_test)
# clf = RandomForestClassifier(n_estimators=200, max_depth=100, random_state=6)
# clf = svm.SVC(gamma=0.0006, C=100000)
# clf = KNeighborsClassifier(n_neighbors=5,p=7)

clf.fit(X_com_train, Y_com_train)
print(clf.score(X_com_test, Y_com_test))


plot_confusion_matrix(Y_com_test, list(clf.predict(X_com_test)),
                      _size=15, classes=v2_names, normalize=True, title='')


X_com_train = np.vstack((x_train, X_multi_fake, X_multi_train))
X_com_test = np.vstack((x_test, X_multi_test))
Y_com_train = np.concatenate((y_train, Y_multi_fake, Y_multi_train))
Y_com_test = np.concatenate((y_test, Y_multi_test))
clf.fit(X_com_train, Y_com_train)
print(clf.score(X_com_test, Y_com_test))
plot_confusion_matrix(Y_com_test, list(clf.predict(X_com_test)),
                      _size=15, classes=v2_names, normalize=True, title='')


X_com_train = np.vstack((x_train, X_multi_train))
X_com_test = np.vstack((x_test, X_multi_test))
Y_com_train = np.concatenate((y_train, Y_multi_train))
Y_com_test = np.concatenate((y_test, Y_multi_test))
clf.fit(X_com_train, Y_com_train)
print(clf.score(X_com_test, Y_com_test))
plot_confusion_matrix(Y_com_test, list(clf.predict(X_com_test)), _size=15,
                      classes=v2_names, normalize=True, title='')

# %%
# ---------------------  grid  --------------------------

time_start = time.time()

param_grid = {"n_estimators": [50, 100, 150, 200, 250, 300],
              "max_depth": [30, 40, 50, 60, 70, 80, 90]
              }
# param_grid = {"gamma":
#               [6e-07, 6e-06, 6e-05, 0.0006,
#                   0.006, 0.06, 0.6, 6, 60, 600],
#               "C": [1e-05, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000],
#               "kernel": ['poly'],
#               "degree" :[11]
#               }
print("Parameters:{}".format(param_grid))

grid_search = GridSearchCV(RandomForestClassifier(),
                           param_grid, cv=5)  # 实例化一个GridSearchCV类
# 训练，找到最优的参数，同时使用最优的参数实例化一个新的SVC estimator。
grid_search.fit(X_com_train, Y_com_train)
print("Test set score:{:.2f}".format(
    grid_search.score(X_com_test, Y_com_test)))
print("Best parameters:{}".format(grid_search.best_params_))
print("Best score on train set:{:.2f}".format(grid_search.best_score_))

# clf = svm.SVC(gamma=0.00006, C=100)
time_end = time.time()
print('time cost', time_end-time_start, 's')


# %%
# precision recall
# %%
# ------------------------ reinforcement ----------------------------
# 什么时候可以到全体 80 以上
# 取一个 data 点出来 T23

# finger[??] trial[0,1,2,3,4,5,6,7,8,9]
queue = [1, 3, 6, 5]
test = [4, 9, 7, 2]

finger_code_list = [24, 20, 18, 12, 10, 6]

X_rein_test = np.empty([0, 30])
for trial in test:
    for index in finger_code_list:
        X_rein_test = np.vstack((
            X_rein_test,
            b_multi[trial][index] -
            np.mean(b_null_multi[trial][index*4:index*4+4], axis=0)
        ))
Y_rein_test = np.array([5, 6, 7, 8, 9, 10]).repeat(20)
Y_rein_test = np.tile(Y_rein_test, len(test))


for finger_index in range(len(finger_code_list)):
    finger = finger_code_list[finger_index]
    print("-----------------------------------")

    X_rein = X_multi_fake
    Y_rein = Y_multi_fake

    if(finger == 12):
        plot_confusion_matrix(Y_rein_test, list(clf.predict(X_rein_test1)),
                              classes=v2_names, normalize=True, title='')
    for index in range(len(queue)):
        step = queue[index]

        print("finger:%d,step:%d" % (finger, step))
        X_rein[finger_index]
        X_rein = np.vstack((
            X_rein,
            b_multi[step][finger] -
            np.mean(b_null_multi[step][finger*4:finger*4+4], axis=0)
        ))
        Y_rein = np.concatenate(
            (Y_rein, np.array([5+finger_index]).repeat(20)))

        # clf = svm.SVC(gamma=0.00002, C=100.)
        clf = KNeighborsClassifier(n_neighbors=10, p=3)

        X_norm = (X_rein - ave_cols) / std_cols
        zero_col_list = np.where(np.isnan(X_norm))[1]
        X_norm = np.delete(X_norm, np.where(np.isnan(X_norm))[1], axis=1)
        print(X_norm.shape)
        X_rein1 = X_norm

        x_test_norm = (X_rein_test - ave_cols) / std_cols
        x_test_norm = np.delete(x_test_norm, np.where(
            np.isnan(x_test_norm))[1], axis=1)
        X_rein_test1 = x_test_norm

        clf.fit(X_rein1, Y_rein)

        print(clf.score(X_rein1, Y_rein), clf.score(X_rein_test1, Y_rein_test))

        v2_names = ['index', 'middle', 'ring', 'little',
                    'thumb', 'T1', 'T2', 'T3', '12', '13', '23']
        v2_names = np.array(v2_names)
        if(finger == 10):
            plot_confusion_matrix(Y_rein_test, list(clf.predict(X_rein_test1)),
                                  classes=v2_names, normalize=True, title='')

    # ---------------------------- training ------------------------
