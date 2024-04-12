from pandas_1 import *

if __name__ == '__main__':
    iris_df = upload_df('../data/lab1/lab1_var1.csv')
    print_df_data(iris_df)
    save_df(iris_df, '../data/lab1/replaced_target_iris.csv')
    show_pair_grid(iris_df, 'label')
    show_histplot(iris_df, 'label', 15)
    show_histplot(iris_df, 'label', 15, kde=True)
