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


# clf = svm.SVC(gamma=0.0002, C=100.)
# clf = svm.SVC(decision_function_shape='ovo')

# Ada boost
# clf = AdaBoostClassifier(n_estimators=100, random_state=0)

# # Random forest
# clf = RandomForestClassifier(n_estimators=150, max_depth=30, random_state=2)

# # KNN
clf = KNeighborsClassifier(n_neighbors=4)


####### training #########
y_train = combine(y_train)
y_test = combine(y_test)

clf.fit(x_train, y_train)
print(clf.score(x_train, y_train), clf.score(x_test, y_test))
v2_names = ['食指', '中指', '无名指', 'little',
            '拇指', '拇指和食指', '拇指和中指', '拇指和无名指', '食指和中指', '食指和无名指', '中指和无名指']
plot_confusion_matrix(y_test, list(clf.predict(x_test)),
                      classes=np.array(v2_names), normalize=True, title=' ', _size=6)
# clf.fit(x_train_pca, y_train)
# print(clf.score(x_test_pca, y_test))
# plot_confusion_matrix(y_test, list(clf.predict(x_test_pca)),
#                       classes=v2_names, normalize=True, title='')

# %%
# other session
# c = [[], ]
# c.append(np.load('cjc_1_single.npy'))
# c.append(np.load('cjc_2_single.npy'))
# c.append(np.load('cjc_3_single.npy'))
# x_test = np.empty([0, 30])
# for trial in range(3, 4):
#     for idx in range(5):
#         x_test = np.vstack(
#             (x_test, c[trial][idx + 1] - np.mean(c[trial][0][idx * 4:idx * 4 + 4], axis=0)))
# y_test = np.array([0, 1, 2, 3, 4])
# y_test = y_test.repeat(20)
# y_test = np.tile(y_test, 1)
# print(clf.score(x_test, y_test))
# plot_confusion_matrix(y_test, list(clf.predict(x_test)),
#                       classes=v2_names, normalize=True, title='')

# %%
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
clf2 = svm.SVC(gamma=0.9, C=0.1)
# clf2 = svm.SVC(gamma=0.00006, C=100)
# clf2 = KNeighborsClassifier(n_neighbors=4, p=3)
clf2.fit(X_multi_train, Y_multi_train)
print(clf2.score(X_multi_test, Y_multi_test))
v2_names = ['食指', '中指', '无名指', 'little',
            '拇指', '拇指和食指', '拇指和中指', '拇指和无名指', '食指和中指', '食指和无名指', '中指和无名指']

v2_names = np.array(v2_names)
plot_confusion_matrix(Y_multi_test, list(clf2.predict(X_multi_test)), _size=10,
                      classes=v2_names, normalize=True, title=' ')
# train-test single & double
# %%
# 图片重叠
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

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
plot_confusion_matrix(Y_multi_test, list(clf.predict(X_multi_test)), _size=8,
                      classes=v2_names, normalize=True, title=' ')

# %%
# ------------------------- comprehensive ---------------------
X_com_train = np.vstack((x_train, X_multi_fake))  # , X_multi_train2))
X_com_test = np.vstack((x_test, X_multi_test))  # , X_multi_test2))
Y_com_train = np.concatenate((y_train, Y_multi_fake))  # , Y_multi_train2))
Y_com_test = np.concatenate((y_test, Y_multi_test))  # , Y_multi_test2))

# Y_com_train = combine(Y_com_train)
# Y_com_test = combine(Y_com_test)
# clf = RandomForestClassifier(n_estimators=200, max_depth=100, random_state=6)
# clf = svm.SVC(gamma=0.00006, C=100)
clf = KNeighborsClassifier(n_neighbors=4)

clf.fit(X_com_train, Y_com_train)
print(clf.score(X_com_test, Y_com_test))


plot_confusion_matrix(Y_com_test, list(clf.predict(X_com_test)),
                      _size=13, classes=v2_names, normalize=True, title=' ')


X_com_train = np.vstack((x_train, X_multi_fake, X_multi_train))
X_com_test = np.vstack((x_test, X_multi_test))
Y_com_train = np.concatenate((y_train, Y_multi_fake, Y_multi_train))
Y_com_test = np.concatenate((y_test, Y_multi_test))
clf.fit(X_com_train, Y_com_train)
print(clf.score(X_com_test, Y_com_test))
# plot_confusion_matrix(Y_com_test, list(clf.predict(X_com_test)),
#                       classes=v2_names, normalize=True, title='')


X_com_train = np.vstack((x_train, X_multi_train))
X_com_test = np.vstack((x_test, X_multi_test))
Y_com_train = np.concatenate((y_train, Y_multi_train))
Y_com_test = np.concatenate((y_test, Y_multi_test))
clf = svm.SVC(gamma=0.00006, C=100)
clf.fit(X_com_train, Y_com_train)
print(clf.score(X_com_test, Y_com_test))
plot_confusion_matrix(Y_com_test, list(clf.predict(X_com_test)), _size=13,
                      classes=v2_names, normalize=True, title=' ')


# %%
# -------------- visualization ---------------------
pca = decomposition.PCA(n_components=2, svd_solver='full')
pca.fit(X_multi_train)
projected = pca.transform(X_com_train)

# 画出每个点的前两个主成份
plt.scatter(projected[:, 0], projected[:, 1], c=Y_com_train+1, edgecolor='none',

            alpha=1, cmap=plt.cm.get_cmap('Spectral', 10))
# plt.xlim((-1000, 1000))
# plt.ylim(ymax=400)
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()


# %%
# ------------------- triple finger trial ----------------------
# 123 | 125 | 135 | 235 | 1235
# 14  | 28  | 26  | 22  | 30
X_tri_fake = np.empty([0, 30])  # [ [],[],...,[] ]
for trial in range(0, 10):  # [2, 3, 4, 5, 7, 8]:
    # combine into new feature
    # b[trial][i] (20*30) + b[trail][j] (20*30) - [(i-1)*4:i*4]
    X_tri_fake = np.vstack((X_tri_fake, b[trial][1] + b[trial][2] + b[trial][3] -
                            np.mean(b[trial][0][0:4], axis=0) -
                            np.mean(b[trial][0][4:8], axis=0) -
                            np.mean(b[trial][0][8:12], axis=0)))
    X_tri_fake = np.vstack((X_tri_fake, b[trial][1] + b[trial][2] + b[trial][5] -
                            np.mean(b[trial][0][0:4], axis=0) -
                            np.mean(b[trial][0][4:8], axis=0) -
                            np.mean(b[trial][0][16:20], axis=0)))
    X_tri_fake = np.vstack((X_tri_fake, b[trial][1] + b[trial][3] + b[trial][5] -
                            np.mean(b[trial][0][0:4], axis=0) -
                            np.mean(b[trial][0][8:12], axis=0) -
                            np.mean(b[trial][0][16:20], axis=0)))
    X_tri_fake = np.vstack((X_tri_fake, b[trial][2] + b[trial][3] + b[trial][5] -
                            np.mean(b[trial][0][4:8], axis=0) -
                            np.mean(b[trial][0][8:12], axis=0) -
                            np.mean(b[trial][0][16:20], axis=0)))
    X_tri_fake = np.vstack((X_tri_fake, b[trial][1] + b[trial][2] + b[trial][3] + b[trial][5] -
                            np.mean(b[trial][0][0:4], axis=0) -
                            np.mean(b[trial][0][4:8], axis=0) -
                            np.mean(b[trial][0][8:12], axis=0) -
                            np.mean(b[trial][0][16:20], axis=0)))
Y_multi2 = np.array([14, 11, 12, 13, 15])
Y_multi2 = Y_multi2.repeat(20)
Y_multi2 = np.tile(Y_multi2, 10)
print(Y_multi2.shape)
X_multi_train2 = X_tri_fake
Y_multi_train2 = Y_multi2

# ----------------------------- load ------------------------
finger_code_list2 = [28, 26, 22, 14, 30]

X_multi3 = np.empty([0, 30])
for index in finger_code_list2:
    for trial in range(0, 10):
        X_multi3 = np.vstack((
            X_multi3,
            b_multi[trial][index] -
            np.mean(b_null_multi[trial][index*4:index*4+4], axis=0)
        ))

print(X_multi3.shape)
# X_multi_train2 = X_multi3[:int(X_multi3.shape[0]/2)]
X_multi_test2 = X_multi3  # [int(X_multi3.shape[0]/2):]

Y_multi_test2 = np.array([11, 12, 13, 14, 15])
Y_multi_test2 = Y_multi_test2.repeat(10*20)
# Y_multi_train2 = np.array([11, 12, 13, 14, 15])
# Y_multi_train2 = Y_multi_train2.repeat(int(len(trail_list)/2)*20)

print(Y_multi_test2.shape)

# X_norm = (X_multi_train2 - ave_cols) / std_cols

# X_norm = np.delete(X_norm, np.where(np.isnan(X_norm))[1], axis=1)
# X_multi_train2 = X_norm

# X_multi_test_norm = (X_multi_test2 - ave_cols) / std_cols
# X_multi_test_norm = np.delete(X_multi_test_norm, np.where(
#     np.isnan(X_multi_test_norm))[1], axis=1)
# X_multi_test2 = X_multi_test_norm


# clf = svm.SVC(gamma=0.0000002, C=100.)
clf = KNeighborsClassifier(n_neighbors=15, p=3)
clf.fit(X_tri_fake, Y_multi_train2)

print(clf.score(X_multi_test2, Y_multi_test2))
v2_names = ['index', 'middle', 'ring', 'little',
            'thumb', 'T1', 'T2', 'T3', '12', '13', '23', 'T12', 'T13', 'T23', '123', 'T123']
v2_names = np.array(v2_names)
plot_confusion_matrix(Y_multi_test2, list(clf.predict(X_multi_test2)),
                      classes=v2_names, normalize=True, title='')


# %%
pca = decomposition.PCA(n_components=2, svd_solver='full')
pca.fit(X_multi_train2)
projected = pca.transform(X_multi_test2)

# 画出每个点的前两个主成份
plt.scatter(projected[:, 0], projected[:, 1], c=Y_multi_test2, edgecolor='none',
            alpha=1, cmap=plt.cm.get_cmap('Spectral', 15))

plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()

# %%
# ------------------------- comprehensive ---------------------
X_com_train = np.vstack((x_train, X_multi_fake))  # , X_multi_train2))
X_com_test = np.vstack((x_test, X_multi_test))  # , X_multi_test2))
Y_com_train = np.concatenate((y_train, Y_multi_fake))  # , Y_multi_train2))
Y_com_test = np.concatenate((y_test, Y_multi_test))  # , Y_multi_test2))

# Y_com_train = combine(Y_com_train)
# Y_com_test = combine(Y_com_test)
# clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=2)
clf = svm.SVC(gamma=0.00002, C=100)
clf = KNeighborsClassifier(n_neighbors=4)

clf.fit(X_com_train, Y_com_train)
print(clf.score(X_com_test, Y_com_test))
v2_names = ['index', 'middle', 'ring', 'little',
            'thumb', 'T1', 'T2', 'T3', '12', '13', '23', 'T12', 'T13', 'T23', '123', 'T123']
v2_names = np.array(v2_names)
plot_confusion_matrix(Y_com_test, list(clf.predict(X_com_test)),
                      classes=v2_names, normalize=True, title='')

# # , X_multi_train2))
X_com_train = np.vstack((x_train, X_multi_train, X_multi_train2))
# # , X_multi_test2))
X_com_test = np.vstack((x_test, X_multi_test, X_multi_test2))
# # , Y_multi_train2))
Y_com_train = np.concatenate((y_train, Y_multi_train, Y_multi_train2))
# # , Y_multi_test2))
Y_com_test = np.concatenate((y_test, Y_multi_test, Y_multi_test2))
# # Y_com_train = combine(Y_com_train)
# # Y_com_test = combine(Y_com_test)

# clf.fit(X_com_train, Y_com_train)
# print(clf.score(X_com_test, Y_com_test))
# plot_confusion_matrix(Y_com_test, list(clf.predict(X_com_test)),
#                       classes=v2_names, normalize=True, title='')

# %%
# ------------------------ reinforcement ----------------------------
# 什么时候可以到全体 80 以上
# 取一个 data 点出来 T23reinfor

# finger[??] trial[0,1,2,3,4,5,6,7,8,9]
queue = [1, 3, 6, 5]
test = [4, 9, 7, 2]

finger_code_list = [24, 20, 18, 12, 10, 6]

X_rein_test = np.empty([0, 15])
for trial in test:
    for index in finger_code_list:
        temp = b_multi[trial][index] - \
            np.mean(b_null_multi[trial][index*4:index*4+4], axis=0)
        temp_norm = (temp - ave_cols) / std_cols
        temp_norm = np.delete(temp_norm, np.where(
            np.isnan(temp_norm))[1], axis=1)
        temp = temp_norm

        X_rein_test = np.vstack((
            X_rein_test, temp
        ))
Y_rein_test = np.array([5, 6, 7, 8, 9, 10]).repeat(20)
Y_rein_test = np.tile(Y_rein_test, len(test))


for finger_index in range(len(finger_code_list)):
    finger = finger_code_list[finger_index]
    print("-----------------------------------")

    X_rein = X_multi_fake
    Y_rein = Y_multi_fake

    if(finger == 10):
        plot_confusion_matrix(Y_rein_test, list(clf.predict(X_rein_test)),
                              classes=v2_names, normalize=True, title='')
    for index in range(len(queue)):
        step = queue[index]

        print("finger:%d,step:%d" % (finger, step))
        # X_rein[finger_index]
        temp = b_multi[step][finger] - \
            np.mean(b_null_multi[step][finger*4:finger*4+4], axis=0)
        temp_norm = (temp - ave_cols) / std_cols
        temp_norm = np.delete(temp_norm, np.where(
            np.isnan(temp_norm))[1], axis=1)
        temp = temp_norm

        X_rein = np.vstack((
            X_rein, temp

        ))
        Y_rein = np.concatenate(
            (Y_rein, np.array([5+finger_index]).repeat(20)))

        # clf = svm.SVC(gamma=0.00002, C=100.)
        # clf = KNeighborsClassifier(n_neighbors=10, p=3)

        # X_norm = (X_rein - ave_cols) / std_cols
        # zero_col_list = np.where(np.isnan(X_norm))[1]
        # X_norm = np.delete(X_norm, np.where(np.isnan(X_norm))[1], axis=1)
        # print(X_norm.shape)
        # X_rein1 = X_norm

        # x_test_norm = (X_rein_test - ave_cols) / std_cols
        # x_test_norm = np.delete(x_test_norm, np.where(
        #     np.isnan(x_test_norm))[1], axis=1)
        # X_rein_test1 = x_test_norm
        # clf = svm.SVC(gamma=0.00006, C=100)
        clf = KNeighborsClassifier(n_neighbors=4)

        clf.fit(X_rein, Y_rein)

        print(clf.score(X_rein, Y_rein), clf.score(X_rein_test, Y_rein_test))

        v2_names = ['index', 'middle', 'ring', 'little',
                    'thumb', 'T1', 'T2', 'T3', '12', '13', '23']
        v2_names = np.array(v2_names)
        if(finger == 10):
            plot_confusion_matrix(Y_rein_test, list(clf.predict(X_rein_test)),
                                  classes=v2_names, normalize=True, title='')

    # ---------------------------- training ------------------------


# %%
feature_list = clf.feature_importances_

b = np.argsort(feature_list)
print((b/3)[::-1])


# %%
# 用两维
X_new_train = X_multi_train[:, 21:30].reshape(-1, 9)
Y_new_train = Y_multi_train

X_new_test = X_multi_test[:, 21:30].reshape(-1, 9)
Y_new_test = Y_multi_test
clf.fit(X_new_train, Y_new_train)
print(clf.score(X_new_test, Y_new_test))
plot_confusion_matrix(Y_new_test, list(clf.predict(X_new_test)),
                      classes=v2_names, normalize=True, title='')
# %%
train_ratio = 0.5
test_ratio = 0.5
finger_code_list = [24, 20, 18, 12, 10, 6]


trail_list_old = [1, 3, 5, 7, 9, 2, 4, 6]
iter_list = list(itertools.permutations(trail_list_old, 8))
iter_result = []
for trail_list in iter_list:
    X_multi = np.empty([0, 30])
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
    # clf = KNeighborsClassifier(n_neighbors=4)
    clf.fit(X_multi_train, Y_multi_train)
    print(clf.score(X_multi_train, Y_multi_train),
          clf.score(X_multi_test, Y_multi_test))
    if (clf.score(X_multi_train, Y_multi_train) > 0.90):
        iter_result.append(trail_list)
    # v2_names = ['index', 'middle', 'ring', 'little',
    #             'thumb', 'T1', 'T2', 'T3', '12', '13', '23']
    # v2_names = np.array(v2_names)
    # plot_confusion_matrix(Y_multi_test, list(clf.predict(X_multi_test)),
    #                     classes=v2_names, normalize=True, title='')

# %%
train_ratio = 0.5
test_ratio = 0.5
finger_code_list = [24, 20, 18, 12, 10, 6]
X_multi = np.empty([0, 30])
for trial in iter_result[620]:
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
# clf = KNeighborsClassifier(n_neighbors=4)
clf.fit(X_multi_train, Y_multi_train)
print(clf.score(X_multi_train, Y_multi_train),
      clf.score(X_multi_test, Y_multi_test))
v2_names = ['index', 'middle', 'ring', 'little',
            'thumb', 'T1', 'T2', 'T3', '12', '13', '23']
v2_names = np.array(v2_names)
plot_confusion_matrix(Y_multi_test, list(clf.predict(X_multi_test)),
                      classes=v2_names, normalize=True, title='')
