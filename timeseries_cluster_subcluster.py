# timeseries_cluster_subcluster.py

'''
This file is effectively a combination of the time_series_clustering.py and
clustered_correlation_matrix.py files also contained in this repo.  The result
is a two-tiered clustering scheme that hierarchically sorts time-series
entries into tier-1 clusters and then tier-2 sub-clusters within the tier-1
clusters.  This approach can be useful for large volumes of time series
data that are expected to have widely-varying structures over their
history, such as sales data for an entire company over several segments
of products.  It provides some great visualizations along the way too.
'''

# Import essential data manipulation packages
import math
import numpy as np
import pandas as pd

# Import clustering utilities
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as spc
from tslearn.clustering import TimeSeriesKMeans

# Import plotting utilities
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# DataFrame `df` structured with rows as observations of a time series,
# 0th column with the names of the times series (e.g., product names, store
# names, etc.), and columns [1:n] the time steps in the series
df

## Crate Tier 1 clusters by time-series
# Set number of clusters to be formed
k_clusters = 9

# Scale each time series' data to have standard mean and variance
scaled_df = df.copy()
for i in range(df.shape[0]):
    scaler = StandardScaler()
    scaled_df.iloc[i, 1:] = scaler.fit_transform(
        df.iloc[i,1:].to_numpy().reshape(-1, 1)
    ).ravel()

# Implement time series clustering
km = TimeSeriesKMeans(n_clusters = k_clusters, metric = 'dtw')
labels = km.fit_predict(scaled_df.iloc[:,1:])

# Add cluster label to DataFrame for later use
df['ts_cluster'] = labels#np.char.zfill(labels.astype(str), 2)

# Set number of plots to be created and initialize subplots
plot_dim = math.ceil(math.sqrt(k_clusters))
fig, axes = plt.subplots(plot_dim, plot_dim, figsize=(20,20))
fig.suptitle('Sales Histories Clustered by Time Series')
row = 0
col = 0

# For each cluster plot every time series with that label
for label in set(labels):
    cluster = []
    for i in range(len(labels)):
        if(labels[i] == label):
            axes[row, col].plot(
                    scaled_df.iloc[i,2:]
                    ,c = "gray"
                    ,alpha = 0.4
                    )
            cluster.append(scaled_df.iloc[i,2:])
            axes[row, col].xaxis.set_major_locator(plt.MaxNLocator(7))
            axes[row, col].tick_params('x', rotation = 45)

    # Plot the average signal
    if len(cluster) > 0:
        axes[row, col].plot(
             np.average(np.vstack(cluster), axis=0)
             ,c = "red"
             )
    axes[row, col].set_title("Cluster " + str(row*plot_dim + col))
    col += 1
    if col%plot_dim == 0:
        row += 1
        col=0

fig.tight_layout()
plt.show()


## Create Tier 2 clusters and provide correlation matrices
# Set up subplots
subclusters = make_subplots(
    rows = plot_dim
    ,cols = plot_dim
    ,subplot_titles = ["Cluster " + str(i) for i in range(k_clusters)]
    ,horizontal_spacing = 0.1/plot_dim
    ,vertical_spacing = 0.12/plot_dim
)

# Create intermediate df to house subcluster ids without interrupting loop
sc_df = df.copy()
sc_df['subcluster_id'] = np.zeros(df.shape[0])

# Add a clustered correlation matrix for each cluster
r = 1; c = 1
for i in range(k_clusters):
    temp_df = df[
        df['ts_cluster'] == i
    ].drop(columns = ['ts_cluster'])

    # Generate intra-cluster correlation matrix
    trans_df = temp_df.transpose()
    cols = trans_df.iloc[0]
    trans_df = trans_df[1:].astype(int)
    trans_df.columns = cols
    corr_mat = trans_df.corr(numeric_only = True).round(3).values

    # Hierarchically cluster the correlation values
    pair_dist = spc.distance.pdist(corr_mat)
    linkage = spc.linkage(pair_dist, method = 'complete')
    ids = spc.fcluster(linkage, 0.5 * pair_dist.max(), 'distance')
    cols = [trans_df.columns.tolist()[i] for i in list((np.argsort(ids)))]
    prod_df_clust = trans_df.reindex(cols, axis = 1)

    # Merge identified subcluster IDs back into the order dataset
    sc_df.loc[df['ts_cluster'] == i, 'subcluster_id'] = ids

    # Print the correlation matrix as a heatmap
    subclusters.add_trace(
        go.Heatmap(
            z = prod_df_clust.corr().round(3)
            ,x = prod_df_clust.corr().round(3).columns
            ,y = prod_df_clust.corr().round(3).columns
            ,name = 'Cluster ' + str(i)
        )
        ,row = r
        ,col = c
    )

    # Iterate to next subplot
    c += 1
    if (c-1)%dim == 0:
        r += 1
        c = 1

# Remove labels and display
subclusters.update_layout(
    height = 1200
    ,title = 'Subcluster Heatmaps'
).update_traces(
    showscale = False
).update_xaxes(
    showticklabels = False
).update_yaxes(
    showticklabels = False
)
subclusters.show()

# Copy subcluster ids back into original df
df['subcluster_id'] = sc_df['subcluster_id']
