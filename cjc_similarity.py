# %%
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import decomposition
from sklearn import datasets
from cjc_util import *
from scipy import stats
# %%

b = []

for index in range(0, 10):
    b.append(np.load('../data/cjc1_%d_single.npy' % index))

b_multi = []
b_null_multi = []
for trial in range(10):
    b_multi.append(np.load('../data/cjc1_%d_multiple.npy' % trial))
    b_null_multi.append(np.load('../data/cjc1_%d_null_multiple.npy' % trial))

# %%
# pierson
finger_code_list = [24, 20, 18, 12, 10, 6]

X = np.empty([0, 30])
for i in range(5):
    for trial in range(10):
        idx = i + 1
        X = np.vstack((X,
                       b[trial][idx] -
                       np.mean(b[trial][0][(idx-1)*4:idx*4], axis=0)
                       ))

X_multi_fake = np.empty([0, 30])  # [ [],[],...,[] ]
Y_multi_fake = np.array([5, 6, 7, 8, 9, 10]).repeat(20)
Y_multi_fake = np.tile(Y_multi_fake, 10)
for trial in range(10):  # [2, 3, 4, 5, 7, 8]:
    for [i, j] in [[5, 1], [5, 2], [5, 3], [1, 2], [1, 3], [2, 3]]:
        # print(i, j)
        # combine into new feature
        # b[trial][i] (20*30) + b[trail][j] (20*30)
        X_multi_fake = np.vstack((X_multi_fake, b[trial][i] + b[trial][j] -
                                  np.mean(b[trial][0][(i-1)*4:i*4], axis=0) -
                                  np.mean(b[trial][0][(j - 1) * 4:j * 4], axis=0)))

X_multi_test = np.empty([0, 30])
Y_multi_test = np.arange(6).repeat(160)
for index in finger_code_list:
    for trial in [1, 2, 3, 4, 5, 6, 7, 9]:

        X_multi_test = np.vstack((
            X_multi_test,
            b_multi[trial][index] -
            np.mean(b_null_multi[trial][index*4: index*4+4], axis=0)
        ))


pccs = np.corrcoef(np.transpose(
    X_multi_fake[20]), np.transpose(X_multi_test[170]))
pccs

# %%
# 各组中心点找出，平均距离找出，合成的数据与目标中心点距离的cdf画出
finger_code_list = [24, 20, 18, 12, 10, 6]

x_ave = np.empty([0, 30])
X = np.empty([0, 30])

for i in range(5):
    for trial in range(10):
        idx = i + 1
        X = np.vstack((X,
                       b[trial][idx] -
                       np.mean(b[trial][0][(idx-1)*4:idx*4], axis=0)
                       ))
    x_ave = np.vstack((x_ave, np.mean(X, axis=0)))

x_fake_ave = np.empty([0, 30])
X_multi_fake = np.empty([0, 30])  # [ [],[],...,[] ]
Y_multi_fake = np.array([6, 7, 8, 9, 10, 11]).repeat(200)
for [i, j] in [[5, 1], [5, 2], [5, 3], [1, 2], [1, 3], [2, 3]]:
    for trial in range(10):  # [2, 3, 4, 5, 7, 8]:
        # combine into new feature
        # b[trial][i] (20*30) + b[trail][j] (20*30)
        X_multi_fake = np.vstack((X_multi_fake, b[trial][i] + b[trial][j] -
                                  np.mean(b[trial][0][(i-1)*4:i*4], axis=0) -
                                  np.mean(b[trial][0][(j - 1) * 4:j * 4], axis=0)))
    x_fake_ave = np.vstack((x_fake_ave, np.mean(X_multi_fake, axis=0)))


x_multi_ave = np.empty([0, 30])
X_multi_test = np.empty([0, 30])
Y_multi_test = np.arange(6).repeat(160)
for index in finger_code_list:
    for trial in [1, 2, 3, 4, 5, 6, 7, 9]:

        X_multi_test = np.vstack((
            X_multi_test,
            b_multi[trial][index] -
            np.mean(b_null_multi[trial][index*4: index*4+4], axis=0)
        ))
    x_multi_ave = np.vstack((x_multi_ave, np.mean(X_multi_test, axis=0)))

# norm_ave = np.vstack((x_ave, x_multi_ave))

# dist = euclidean_distances(norm_ave)


# %%
# column_lst = [5, 6, 7, 8, 9, 10]

# for col, lst in zip(column_lst, unstrtf_lst):
# for i in range(6):
#     pccs = np.corrcoef(x_multi_ave[5], x_fake_ave[i])
#     print(pccs)


# %%
num = dist.sum(axis=1)
den = 10
dist_ave = num/den

# %%
for x in range(6):
    print(euclidean_distances(
        x_fake_ave[x, :].reshape(1, 30), x_multi_ave[x, :].reshape(1, 30)))

# %%
# -------  双指 和别人的距离
dist = np.empty([11, 0])
for x in range(6):
    dist = np.hstack((dist, euclidean_distances(
        norm_ave, x_multi_ave[x, :].reshape(1, 30))))
num = dist.sum(axis=0)
den = 10
print(num / den)
dist.max(axis=0)
# %%

# f = {0: 2, 1: 4, 2: 5, 3: 0, 4: 1, 5: 3}
fig = plt.figure(figsize=(12, 6))
axes_array = []
Mean = [122, 90, 137, 82, 81, 81]
Max = [229, 171, 253, 138, 146, 136]
title = ['拇指和食指', '拇指和中指', '拇指和无名指', '食指和中指', '食指和无名指', '中指和无名指']
for i in range(6):
    axes_array.append(fig.add_subplot(2, 3, i+1))

for x in range(6):
    data = euclidean_distances(
        X_multi_fake[x*200:x*200+200], x_multi_ave[x, :].reshape(1, 30)).reshape(1, -1)[0].tolist()

    data_size = len(data)

    # Set bins edges
    data_set = sorted(set(data))
    bins = np.append(data_set, data_set[-1]+1)

    # Use the histogram function to bin the data
    counts, bin_edges = np.histogram(data, bins=bins, density=False)

    counts = counts.astype(float)/data_size

    # Find the cdf
    cdf = np.cumsum(counts)

    axes_array[x].plot(bin_edges[0:-1], cdf, linestyle='--',
                       marker="o", color='b')
    axes_array[x].vlines(Mean[x], 0, 1.0, colors="g", linestyles="dashed")
    axes_array[x].vlines(Max[x], 0, 1.0, colors="r", linestyles="dashed")
    # axes_array[x].set_ylim((0, 1))
    axes_array[x].set(
        ylabel='CDF',
        xlabel='欧式距离',
        title=title[x])
    # plt.grid(True)
    # Plot the cdf
    # plt.plot(bin_edges[0:-1], cdf, linestyle='--', marker="o", color='b')
    # plt.ylim((0, 1))
    # plt.ylabel("CDF")
    # plt.grid(True)
# plt.annotate(fontsize=20)
plt.rcParams.update({'font.size': 20})
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.tight_layout()
plt.show()
# # Choose how many bins you want here
# num_bins = 600

# # Use the histogram function to bin the data
# counts, bin_edges = np.histogram(data, bins=num_bins, normed=True)

# # Now find the cdf
# cdf = np.cumsum(counts)

# # And finally plot the cdf
# plt.plot(bin_edges[1:], cdf)

# plt.show()
# %%
data = euclidean_distances(
    X_multi_test[0:160], x_multi_ave[4, :].reshape(-1, 30)).reshape(1, -1)[0].tolist()
data_size = len(data)

# Set bins edges
data_set = sorted(set(data))
bins = np.append(data_set, data_set[-1]+1)

# Use the histogram function to bin the data
counts, bin_edges = np.histogram(data, bins=bins, density=False)

counts = counts.astype(float)/data_size

# Find the cdf
cdf = np.cumsum(counts)

# Plot the cdf
plt.plot(bin_edges[0:-1], cdf, linestyle='--', marker="o", color='b')
plt.ylim((0, 1))
plt.ylabel("CDF")
plt.grid(True)

plt.show()
# %%
x_com = np.vstack((X_multi_fake, X_multi_test))
y_com = np.concatenate((Y_multi_fake, Y_multi_test))
pca = decomposition.PCA(n_components=2, svd_solver='full')
pca.fit(x_com)
projected = pca.transform(x_com)

# 画出每个点的前两个主成份
plt.scatter(projected[:, 0], projected[:, 1], c=y_com+1, edgecolor='none',

            alpha=1, cmap=plt.cm.get_cmap('Spectral', 12))
# plt.xlim((377864100, 377864200))
# # plt.ylim((-750,500))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()


# %%
x_com_ave = np.vstack((x_fake_ave, x_multi_ave))
projected = pca.transform(x_com_ave)
plt.scatter(projected[:, 0], projected[:, 1], c=np.array([1,2,3,4,5,6,7,8,9,10,11,12]), edgecolor='none',

            alpha=1, cmap=plt.cm.get_cmap('Spectral', 12))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()


# %%
