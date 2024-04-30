import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from lab1.pandas_1 import upload_df, print_df_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt


def split_and_show(xp_split, yp_split, test_size=0.3, shuffle=False):
    xp_train, xp_test, yp_train, yp_test = train_test_split(xp_split, yp_split, test_size=test_size, shuffle=shuffle)
    print(xp_test)
    print(xp_train)
    print(yp_test)
    print(yp_train)
    for title in xp_split.columns.tolist():
        plt.scatter(xp_train[title], yp_train, c='b', marker='.')
        plt.scatter(xp_test[title], yp_test, c='r', marker='.')
        plt.grid()
        plt.show()
    return xp_train, xp_test, yp_train, yp_test


def metrics(y_real, y_predicted):
    print(r2_score(y_real, y_predicted))
    print(mean_squared_error(y_real, y_predicted))
    print(mean_absolute_error(y_real, y_predicted))


def make_linear_regression(xp_train,  yp_train):
    lin_reg2 = LinearRegression()
    lin_reg2.fit(xp_train, yp_train)
    print("Полученные коэффициенты при x ", lin_reg2.coef_, "Полученный свободный член ", lin_reg2.intercept_)
    return lin_reg2


def make_lasso_regression(xp_train, yp_train):
    lasso1 = Lasso(0.05)
    lasso1.fit(xp_train, yp_train)
    print("Полученные коэффициенты при x ", lasso1.coef_, "Полученный свободный член ", lasso1.intercept_)
    return lasso1


def show_all_metrics(yp, xp, linear_reg2, text):
    print(text)
    metrics(yp,
            linear_reg2.coef_[0] * xp["x1"] + linear_reg2.coef_[1] * xp["x2"] + linear_reg2.coef_[2] *
            xp["x3"] + linear_reg2.coef_[3] * xp["x5"] + linear_reg2.intercept_)


if __name__ == '__main__':
    data = upload_df('../data/lab3/lab3_lin3.csv', delete_first=False)
    print_df_data(data)
    xp_train, xp_test, yp_train, yp_test = split_and_show(data[["x1", "x2", "x3", "x5"]], data["y"])

    linear_reg2 = make_linear_regression(xp_train, yp_train)
    show_all_metrics(yp_train, xp_train, linear_reg2, "========== train metrics ==========")
    show_all_metrics(yp_test, xp_test, linear_reg2, "========== text metrics ==========")

    lasso1 = make_lasso_regression(xp_train, yp_train)
    show_all_metrics(yp_train, xp_train, lasso1, "========== train metrics ==========")
    show_all_metrics(yp_test, xp_test, lasso1, "========== text metrics ==========")
