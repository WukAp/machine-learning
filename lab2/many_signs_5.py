import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from lab1.dim_reduction_5 import get_reduced_dimensionality_df
from lab1.pandas_1 import upload_df, show_pair_grid
import seaborn as sns

from lab2.prepearing_1 import scale

from lab2.k_means_2 import show_elbow_method, show_silhouette_method


def k_mean_clusterize(df, clusters):
    norm_km = KMeans(n_clusters=clusters)
    norm_clust = norm_km.fit_predict(df)
    return norm_clust


def dbscan_clusterize(df, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_clust = dbscan.fit_predict(df)
    return dbscan_clust


def ag_clusterize(df, clusters=5, linkage='ward'):
    agc = AgglomerativeClustering(n_clusters=clusters, linkage=linkage)
    ag_clust = agc.fit_predict(df)
    return ag_clust


def show_bscan_clusterize(df, eps, min_samples):
    cluster_df = df
    cluster_df['cluster'] = dbscan_clusterize(df, eps, min_samples)
    print(cluster_df)
    show_pair_grid(cluster_df, 'cluster')  # долго работает


def show_k_mean_clusterize(df, cluster):
    cluster_df = df
    cluster_df['cluster'] = k_mean_clusterize(df, clusters=cluster)
    print(cluster_df)
    show_pair_grid(cluster_df, 'cluster')  # долго работает


def show_ag_clusterize(df, cluster):
    cluster_df = df
    cluster_df['cluster'] = ag_clusterize(df, clusters=cluster)
    print(cluster_df)
    show_pair_grid(cluster_df, 'cluster')


if __name__ == '__main__':
    columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
               'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
               'pH', 'sulphates', 'alcohol', 'q']

    df_uploaded = scale(upload_df('../data/lab2/lab2_winequality_red.csv', delete_first=False), columns)
    df_uploaded.drop(df_uploaded.columns[[-1]], axis=1, inplace=True)

    df_dim2 = get_reduced_dimensionality_df(df_uploaded, TSNE(n_components=2), ['x', 'y'])
    # show_elbow_method(df_uploaded, y=15)  # 10-12
    # show_silhouette_method(df_uploaded, y=15) #10

    # show_k_mean_clusterize(df_uploaded, 10)

    # show_bscan_clusterize(df_uploaded, 0.1, 12)
    # show_ag_clusterize(df_uploaded, 10)

    # df_dim2['cluster'] = k_mean_clusterize(df_uploaded, clusters=18)
    # sns.scatterplot(df_dim2, x='x', y='y', hue='cluster', palette='tab10')
    # plt.show()

    df_dim2['cluster'] = dbscan_clusterize(df_uploaded, 0.17, 5)
    sns.scatterplot(df_dim2, x='x', y='y', hue='cluster', palette='tab10')
    plt.show()

    df_dim2['cluster'] = ag_clusterize(df_uploaded, clusters=10)
    sns.scatterplot(df_dim2, x='x', y='y', hue='cluster', palette='tab10')
    plt.show()
