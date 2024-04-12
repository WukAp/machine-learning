import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

class_mapping = {
    0: 'Iris-setosa',
    1: 'Iris-versicolor',
    2: 'Iris-virginica'
}


def upload_df(file_name, delete_first=True):
    df = pd.read_csv(file_name, delimiter=',')
    if delete_first:
        return df.drop(df.columns[[0]], axis=1)
    else:
        return df


def save_df(df, file_name):
    df.to_csv(file_name, index=False)


def print_df_data(df):
    print("\n============== head ==============")
    print(df.head())
    print("\n============= describe ===============")
    print(df.describe())


def replace_df_column(df, column, class_mapping):
    df_copy = df.copy(deep=True)
    df_copy[column] = df_copy['target'].replace(class_mapping)
    print("\n============= replaced  ===============")
    print(df_copy.head())
    return df_copy


def show_pair_grid(df, hue):
    sns.set_theme(style="white")
    g = sns.PairGrid(df, diag_sharey=False, hue=hue)
    g.map_upper(sns.scatterplot, s=15)
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot)
    g.add_legend()
    sns.set_palette("Set1")
    plt.show()


def show_histplot(df, hue, bins_in_histogram=15, row=2, column=2, multiple='layer', kde=False):
    column_names = df.columns.values
    fig, axes = plt.subplots(row, column, figsize=(18, 10))
    for i in range(0, min(column_names.size, row * column)):
        sns.histplot(df,
                     ax=axes[i // column, i % column],
                     x=df[column_names[i]],
                     element='step',
                     hue=hue,
                     kde=kde,
                     bins=bins_in_histogram,
                     multiple=multiple)
    sns.set_palette("Set1")
    plt.show()


if __name__ == '__main__':
    iris_df = upload_df('../data/lab1/iris.csv')
    print_df_data(iris_df)
    iris_df = replace_df_column(iris_df, 'target', class_mapping)
    save_df(iris_df, '../data/lab1/replaced_target_iris.csv')
    show_pair_grid(iris_df, 'target')
    for i in range(5, 35, 5):
        show_histplot(iris_df, 'target', i)
    show_histplot(iris_df, 'target', 15, multiple='stack')
    show_histplot(iris_df, 'target', 15)
    show_histplot(iris_df, 'target', kde=True)
