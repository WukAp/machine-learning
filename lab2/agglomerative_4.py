import pandas as pd
import seaborn as sns
import scipy.cluster.hierarchy as shc
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering

from lab2.prepearing_1 import scale, get_first_df, get_second_df, get_third_df


def clusterize(df, clusters=5, linkage='average'):
    agc = AgglomerativeClustering(n_clusters=clusters, linkage=linkage)
    ag_clust = agc.fit_predict(df)
    norm_ag = pd.DataFrame(df, columns=['x', 'y'])
    norm_ag['cluster'] = ag_clust
    return norm_ag, ag_clust


def show_clusters(df, clusters=5, linkage='average', name="===================="):
    print(name)
    norm_ag, ag_clust = clusterize(df.to_numpy(), clusters, linkage)
    print(set(ag_clust))

    shc.dendrogram(shc.linkage(df, method=linkage))
    plt.show()

    sns.scatterplot(norm_ag, x='x', y='y', hue='cluster', palette='tab10')
    plt.show()


if __name__ == '__main__':
    first_df = scale(get_first_df())
    second_df = scale(get_second_df())
    third_df = scale(get_third_df())

    show_clusters(first_df, name="======== lab2_blobs ========", clusters=5, linkage='ward')
    show_clusters(first_df, name="======== lab2_blobs ========", clusters=5, linkage='average')
    show_clusters(first_df, name="======== lab2_blobs ========", clusters=5, linkage='complete')
    show_clusters(first_df, name="======== lab2_blobs ========", clusters=5, linkage='single')

    show_clusters(second_df, name="======== lab2_checker ========", clusters=5, linkage='ward')
    show_clusters(second_df, name="======== lab2_checker ========", clusters=5, linkage='average')
    show_clusters(second_df, name="======== lab2_checker ========", clusters=5, linkage='complete')
    show_clusters(second_df, name="======== lab2_checker ========", clusters=5, linkage='single')

    #show_clusters(third_df, name="======== lab2_noisymoons ========", clusters=2, linkage='ward')
    #show_clusters(third_df, name="======== lab2_noisymoons ========", clusters=2, linkage='average')

    show_clusters(third_df, name="======== lab2_noisymoons ========", clusters=5, linkage='ward')
    show_clusters(third_df, name="======== lab2_noisymoons ========", clusters=5, linkage='average')
    show_clusters(third_df, name="======== lab2_noisymoons ========", clusters=5, linkage='complete')
    show_clusters(third_df, name="======== lab2_noisymoons ========", clusters=7, linkage='single')
