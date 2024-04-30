from lab1.pandas_1 import upload_df, print_df_data

if __name__ == '__main__':
    data2 = upload_df('../data/lab3/lab3_poly2.csv', delete_first=False)
    print_df_data(data2)