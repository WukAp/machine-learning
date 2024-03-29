import matplotlib.pyplot as plt

from pandas_1 import upload_df, replace_df_column
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd


def get_reduced_dimensionality_df(df, transformer):
    a_dim_2 = transformer.fit_transform(df)
    return pd.DataFrame(
        data=a_dim_2,
        columns=['x', 'y'])


def show_reduced_dimensionality(df, transformer):
    df_dim_2 = get_reduced_dimensionality_df(df, transformer)

    class_mapping = {
        0: 'Iris-setosa',
        1: 'Iris-versicolor',
        2: 'Iris-virginica'
    }
    x = replace_df_column(df, 'target', class_mapping)
    df_dim_2['target'] = x['target']

    sns.scatterplot(
        x='x',
        y='y',
        data=df_dim_2,
        hue='target',
        legend=True
    )
    plt.show()


def show_PCA(df):
    show_reduced_dimensionality(df, PCA(n_components=2))


def show_TSNE(df):
    show_reduced_dimensionality(df, TSNE(n_components=2))


if __name__ == '__main__':
    iris_df = upload_df('../data/iris.csv')
    show_PCA(iris_df)
    show_TSNE(iris_df)
