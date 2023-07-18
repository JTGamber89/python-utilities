# Time Series Clustering

# Import essential data manipulation packages
import math
import numpy as np
import pandas as pd

# Import data science utilities
from sklearn.preprocessing import StandardScaler
from tslearn.clustering import TimeSeriesKMeans

# Import plotting utilities
import matplotlib.pyplot as plt

df = pd.read_csv('./file-path.csv')

# Set number of clusters to be formed
k = 12

# Scale to have standard mean and variance
scaled_df = df.copy()
for i in range(order_df.shape[0]):
    scaler = StandardScaler()
    scaled_df.iloc[i, 1:] = scaler.fit_transform(
        order_df.iloc[i,1:].to_numpy().reshape(-1, 1)
    ).reshape(1, -1)

# Implement time series clustering using dynamic time warping
km = TimeSeriesKMeans(n_clusters = k, metric = "dtw")
labels = km.fit_predict(scaled_df.iloc[:,1:])

# Set number of plots to be created and initialize subplots
plot_num = math.ceil(math.sqrt(k))
fig, axes = plt.subplots(plot_num-1, plot_num, figsize=(25,25))
fig.suptitle('Clusters')
row = 0
col = 0

# For each cluster plot every history with that label
for label in set(labels):
    cluster = []
    for i in range(len(labels)):
            if(labels[i] == label):
                axes[row, col].plot(
                     scaled_df.iloc[i,1:]
                     ,c="gray"
                     ,alpha=0.4
                     )
                cluster.append(scaled_df.iloc[i,1:])

    # Plot the average signal
    if len(cluster) > 0:
        axes[row, col].plot(
             np.average(np.vstack(cluster),axis=0)
             ,c="red"
             )
    axes[row, col].set_title("Cluster " + str(row + col))
    col += 1
    if col%plot_num == 0:
        row += 1
        col=0

plt.show()
