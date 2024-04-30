from lab1.pandas_1 import upload_df, print_df_data
from sklearn.model_selection import train_test_split
if __name__ == '__main__':
    data = upload_df('../data/lab3/lab3_lin3.csv', delete_first=False)
    print_df_data(data)
    xp_split = data[["x1","x2","x3","x5"]]
    yp_split = data["y"]
    xp_train, xp_test, yp_train, yp_test = train_test_split(xp_split, yp_split, test_size=0.3, shuffle=False)
    print(xp_test, xp_train, yp_test, yp_train)