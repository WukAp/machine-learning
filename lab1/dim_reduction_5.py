import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

from lab1.pandas_1 import replace_df_column, upload_df


def get_reduced_dimensionality_df(df, transformer, column):
    a_dim_2 = transformer.fit_transform(df)
    return pd.DataFrame(
        data=a_dim_2,
        columns=column)


def show_reduced_dimensionality(df, transformer):
    df_dim_2 = get_reduced_dimensionality_df(df, transformer, ['x', 'y'])

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
    iris_df = upload_df('../data/lab1/iris.csv')
    show_PCA(iris_df)
    show_TSNE(iris_df)
