import numpy as np
import scipy.cluster.hierarchy as spc
import plotly.express as px

# Pivot the dataframe to wide_format (if necessary) such that
# columns are features and rows are observations
df

# Generate the initial correlation matrix
corr_mat = df.corr().round(3).values

# Hierarchically cluster the correlation values
pair_dist = spc.distance.pdist(corr_mat)
linkage = spc.linkage(pair_dist, method = 'complete')
ids = spc.fcluster(linkage, 0.5 * pair_dist.max(), 'distance')
cols = [df.columns.tolist()[i] for i in list((np.argsort(ids)))]
df_clust = df.reindex(cols, axis = 1)

# Print the correlation matrix as a heatmap
px.imshow(
    df_clust.corr().round(3)
    ,text_auto = True
    ,title = '[Some cool title]'
)
