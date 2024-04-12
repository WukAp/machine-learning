import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from lab1.pandas_1 import upload_df, print_df_data


def get_first_df():
    return upload_df('../data/lab2/lab2_blobs.csv', delete_first=False)


def get_second_df():
    return upload_df('../data/lab2/lab2_checker.csv', delete_first=False)


def get_third_df():
    return upload_df('../data/lab2/lab2_noisymoons.csv', delete_first=False)


def show_scatterplot(df):
    sns.scatterplot(df, x='x', y='y')
    plt.show()


def scale(df, columns):
    arr = df.to_numpy()
    scaler = MinMaxScaler()
    scaler.fit(arr)
    arr = scaler.transform(arr)
    return pd.DataFrame(arr, columns=columns)


def show_all(df, name):
    print(name)

    print_df_data(df)
    show_scatterplot(df)

    df = scale(df, ['x', 'y'])
    show_scatterplot(df)


if __name__ == '__main__':
    show_all(get_first_df(), 'lab2_blobs.csv')
    show_all(get_second_df(), 'lab2_checker.csv')
    show_all(get_third_df(), 'lab2_noisymoons.csv')
