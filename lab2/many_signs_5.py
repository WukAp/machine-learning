import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import scipy.cluster.hierarchy as shc
from lab1.dim_reduction_5 import get_reduced_dimensionality_df
from lab1.pandas_1 import upload_df, show_pair_grid
import seaborn as sns

from lab2.prepearing_1 import scale

from lab2.k_means_2 import show_elbow_method, show_silhouette_method, show_clusters


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


def get_dim2_plot(df, cluster):
    df_dim2 = get_reduced_dimensionality_df(df, PCA(n_components=2), ['x', 'y'])
    df_dim2['cluster'] = cluster

    print(df)
    sns.scatterplot(
        x='x',
        y='y',
        data=df_dim2,
        hue='cluster',
        legend=True,
        palette='tab10'
    )
    plt.show()
    df_dim2 = get_reduced_dimensionality_df(df, TSNE(n_components=2), ['x', 'y'])
    df_dim2['cluster'] = cluster

    print(df)
    sns.scatterplot(
        x='x',
        y='y',
        data=df_dim2,
        hue='cluster',
        legend=True,
        palette='tab10'
    )
    plt.show()


def show_dbscan_clusterize(df, eps, min_samples, cluster_name):
    cluster_df = df.copy()
    clusters_column = dbscan_clusterize(df, eps, min_samples)
    cluster_df['cluster'] = clusters_column

    get_dim2_plot(df, clusters_column)
    # show_boxplot(cluster_df, cluster_name)


def show_k_mean_clusterize(df, cluster, cluster_name):
    cluster_df = df.copy()
    clusters_column = k_mean_clusterize(df, clusters=cluster)
    cluster_df['cluster'] = clusters_column
    get_dim2_plot(df, clusters_column)
    #show_pair_grid(cluster_df, 'cluster')
    show_boxplot(cluster_df, cluster_name)


def show_ag_clusterize(df, cluster_name, cluster=5, linkage = 'ward'):
    cluster_df = df.copy()

    clusters_column = ag_clusterize(df,  cluster, linkage=linkage)
    cluster_df['cluster'] = clusters_column
    get_dim2_plot(df, clusters_column)
    shc.dendrogram(shc.linkage(df, method=linkage))
    plt.show()
    #show_boxplot(cluster_df, cluster_name)
    #show_pair_grid(cluster_df, 'cluster')
    for i in range(0, 6):
        tmp_df = cluster_df.loc[(cluster_df['cluster'] == i)]
        print("claster == ",  i)
        print(tmp_df.describe())
    print(cluster_df.describe())



def show_boxplot(norm_pd, column):
    for i in column:
        sns.violinplot(norm_pd, x=i, hue='cluster', palette='tab10', )
        plt.title = i
        plt.show()


if __name__ == '__main__':
    columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
               'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
               'pH', 'sulphates', 'alcohol', 'quality']

    df_uploaded = scale(upload_df('../data/lab2/lab2_winequality_red.csv', delete_first=False), columns)
    #show_pair_grid(df_uploaded, hue='quality')
    df_uploaded.drop(df_uploaded.columns[[-1]], axis=1, inplace=True)

    #show_elbow_method(df_uploaded, y=10)  # 10-12
    #show_silhouette_method(df_uploaded, y=15) #10
    columns = columns[0: -1]

    #show_k_mean_clusterize(df_uploaded, 6, columns)  # норм
    #show_k_mean_clusterize(df_uploaded, 4, columns)  # норм
    #show_k_mean_clusterize(df_uploaded, 8, columns)  # норм
    show_ag_clusterize(df_uploaded,  columns, 6)
    #show_ag_clusterize(df_uploaded, 7, columns)
    #show_ag_clusterize(df_uploaded, 5, columns)
    #show_dbscan_clusterize(df_uploaded, 0.2, 3, columns)
