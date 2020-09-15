import sklearn
from sklearn.cluster import MiniBatchKMeans
from itertools import cycle, islice
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

kMeans = MiniBatchKMeans(n_clusters=16)  

dataset_path = "datasets/3.csv"
dataset = pd.read_csv(dataset_path)
X = dataset[['in2', 'out1', 'out2']].values
kMeans.fit_predict(X)
y_pred = kMeans.labels_
colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_pred) + 1))))

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:, 0], X[:, 1], X[:, 2], color=colors[y_pred])
plt.show()