from lab1.pandas_1 import upload_df, print_df_data, replace_df_column
import seaborn as sns
import matplotlib.pyplot as plt

from lab3.linear_regression_1 import split_and_show


def show_scatterplot(data):
    sns.scatterplot(data, x='X1', y='X2', hue='Class', palette='tab10')
    plt.show()


def get_split_data():
    data = upload_df('../data/lab4/lab4_6.csv', delete_first=False)

    data = replace_df_column(data, 'Class', {
        'MO': 0,
        'ON': 1,
    })
    print_df_data(data)
    show_scatterplot(data)
    return data[["X1", "X2"]], data["Class"]


if __name__ == '__main__':
    data = upload_df('../data/lab4/lab4_6.csv', delete_first=False)
    print_df_data(data)
    show_scatterplot(data)
    x_train, x_test, y_train, y_test = split_and_show(data[["X1", "X2"]], data["Class"], alpha=0.2)
