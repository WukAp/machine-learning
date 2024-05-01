import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
import seaborn as sns

from lab2.prepearing_1 import get_first_df, get_second_df, get_third_df, scale
from sklearn.metrics import silhouette_score


def show_elbow_method(df, y=15):
    inert_list = []
    for i in range(y):
        temp_km = KMeans(i + 1, n_init=5)
        temp_km.fit(df)
        inert_list.append(temp_km.inertia_)

    plt.plot(list(range(1, y + 1)), inert_list)
    plt.ylabel('Инерция')
    plt.xlabel('Количество кластеров')
    plt.xticks(list(range(1, y + 1)))
    plt.grid()
    plt.show()


def show_silhouette_method(df, y=15):
    sil_list = []
    for i in range(1, y):
        temp_km = KMeans(n_clusters=i+1,  n_init=5)
        temp_clust = temp_km.fit_predict(df)
        sil_list.append(silhouette_score(df, temp_clust))

    plt.plot(list(range(2, y + 1)), sil_list)
    plt.ylabel('Среднее значение коэффициента силуэта')
    plt.xlabel('Количество кластеров')
    plt.xticks(list(range(1, y + 1)))
    plt.grid()
    plt.show()


def show_voronoi_diagram(a, norm_km):
    h = 0.02
    x_min, x_max = a[:, 0].min() - 0.5, a[:, 0].max() + 0.5
    y_min, y_max = a[:, 1].min() - 0.5, a[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    z_clust = norm_km.predict(np.c_[xx.ravel(), yy.ravel()])
    z_clust = z_clust.reshape(xx.shape)
    norm_cent = norm_km.cluster_centers_
    plt.imshow(
        z_clust,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
    )  # рисуем области
    plt.plot(a[:, 0], a[:, 1], "k.", markersize=4)  # рисуем точки
    plt.scatter(
        norm_cent[:, 0],
        norm_cent[:, 1],
        marker="x",
        s=169,
        linewidths=3,
        color="w",
        zorder=10,
    )
    plt.show()


def show_scatterplot(norm_pd):
    sns.scatterplot(norm_pd, x='x', y='y', hue='cluster', palette='tab10')
    plt.show()


def show_boxplot(norm_pd):
    sns.boxplot(norm_pd, x='x', hue='cluster', palette='tab10')
    plt.show()

    sns.boxplot(norm_pd, x='y', hue='cluster', palette='tab10')
    plt.show()


def clusterize(df, clusters):
    norm_km = KMeans(n_clusters=clusters)
    norm_clust = norm_km.fit_predict(df)
    norm_pd = pd.DataFrame(df, columns=['x', 'y'])
    norm_pd['cluster'] = norm_clust
    return norm_pd, norm_km


def show_clusters(df, clusters, name="===================="):
    norm_pd, norm_km = clusterize(df, clusters)
    show_scatterplot(norm_pd)
    show_voronoi_diagram(df.to_numpy(), norm_km)
    show_boxplot(norm_pd)
    print(name)
    for i in range(clusters):
        tmp_df = norm_pd.loc[(norm_pd['cluster'] == i)]
        print("claster == ",  i)
        print(tmp_df.describe())


if __name__ == '__main__':
    first_df = scale(get_first_df())
    second_df = scale(get_second_df())
    third_df = scale(get_third_df())



    show_clusters(first_df, 5, "======== lab2_blobs ========")
    show_clusters(second_df, 5, "======== lab2_checker ========")
    show_clusters(third_df, 8, "======== lab2_noisymoons ========")
