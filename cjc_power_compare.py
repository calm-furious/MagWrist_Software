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

a = []

for index in range(0, 10):
    a.append(np.load('../data/cjc_%d_single.npy' % index))

b = []

for index in range(0, 10):
    b.append(np.load('../data/cjc1_%d_single.npy' % index))

c = []

index = 0
c.append(np.load('../data/abc_%d_single.npy' % index))

# %%
a_ave = []
for i in range(5):
    X = np.empty([0, 30])
    for trial in range(10):
        idx = i + 1
        X = np.vstack((X,
                       a[trial][idx] -
                       np.mean(a[trial][0][(idx-1)*4:idx*4], axis=0)
                       ))
    # print(np.mean(X), np.std(X))
    a_ave.append(np.mean(X))


b_ave = []
for i in range(5):
    X = np.empty([0, 30])
    for trial in range(10):
        idx = i + 1
        X = np.vstack((X,
                       np.abs(b[trial][idx] -
                              np.mean(b[trial][0][(idx-1)*4:idx*4], axis=0))
                       ))
    print(np.argmax(X, axis=1))
    print(np.mean(X[:,  np.argmax(X, axis=1)]))
    b_ave.append(np.mean(X[:,  np.argmax(X, axis=1)]))
print("------------------------")
c_ave = []
for i in range(5):
    X = np.empty([0, 30])
    trial = 0
    idx = i + 1
    X = np.vstack((X,
                   np.abs(c[trial][idx] -
                          np.mean(c[trial][0][(idx-1)*4:idx*4], axis=0))
                   ))
    print(np.mean(X[:, np.argmax(X, axis=1)]))
    c_ave.append(np.mean(X[:, np.argmax(X, axis=1)]))
# %%
b_ave = []

for i in range(5):
    X = np.empty([0, 30])
    for trial in range(10):
        idx = i + 1
        X = np.vstack((X,
                       np.abs(b[trial][idx] -
                              np.mean(b[trial][0][(idx-1)*4:idx*4], axis=0))
                       ))
    b_ave.append(np.mean(X**2))
print("------------------------")
c_ave = []

for i in range(5):
    X2 = np.empty([0, 30])
    trial = 0
    idx = i + 1
    X2 = np.vstack((X2,
                    np.abs(c[trial][idx] -
                           np.mean(c[trial][0][(idx-1)*4:idx*4], axis=0))
                    ))
    c_ave.append(np.mean(X2**2))

# %%
