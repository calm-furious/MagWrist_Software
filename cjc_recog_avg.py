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
# b1 = np.load('cjc_1_single.npy')

# %%
# pre processing

