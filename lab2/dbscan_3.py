import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN

from lab2.prepearing_1 import get_first_df, get_second_df, get_third_df, show_scatterplot, scale
import seaborn as sns


def clusterize(df, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_clust = dbscan.fit_predict(df)
    norm_pd = pd.DataFrame(df, columns=['x', 'y'])
    norm_pd['cluster'] = dbscan_clust
    return norm_pd, dbscan_clust


def show_clusters(df, eps, min_samples, name="===================="):
    print(name)
    norm_pd, _ = clusterize(df.to_numpy(), eps, min_samples)

    sns.scatterplot(norm_pd, x='x', y='y', hue='cluster', palette='tab10')
    plt.show()


if __name__ == '__main__':
    first_df = scale(get_first_df(),columns=['x', 'y'])
    second_df = scale(get_second_df(), columns=['x', 'y'])
    third_df = scale(get_third_df(), columns=['x', 'y'])

    show_clusters(first_df, name="======== lab2_blobs ========", eps=0.11, min_samples=5)
    show_clusters(second_df, name="======== lab2_checker ========", eps=0.1, min_samples=30)
    show_clusters(third_df, name="======== lab2_noisymoons ========", eps=0.112, min_samples=15)
