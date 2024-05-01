import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import tensorflow as tf
from lab1.pandas_1 import upload_df, print_df_data
from lab3.linear_regression_1 import split_and_show, show_predicted_plot, \
    make_regression, show_all_metrics


def get_predicted_y(xp, reg):
    return reg.coef_[0] * xp["x"] + reg.intercept_
def show_poly_plot(n):
    x_line_nl = np.linspace(-1.3, 1.3, 100)
    poly_transform = PolynomialFeatures(n)
    xnl_poly = poly_transform.fit_transform(xp_train)
    lin_nl2 = LinearRegression(fit_intercept=False)
    lin_nl2.fit(xnl_poly, yp_train)
    print(lin_nl2.coef_)
    poly_line = poly_transform.transform(x_line_nl.reshape(-1, 1))
    y_pred2 = lin_nl2.predict(poly_line)
    plt.plot(xp_train, yp_train, 'b.')
    plt.plot(xp_test, yp_test, 'r.')
    plt.plot(x_line_nl, y_pred2, 'k-')
    plt.legend(('train', 'test', 'predict'))
    plt.title("PolynomialFeatures " + str(n))
    plt.grid()
    plt.show()

if __name__ == '__main__':
    data = upload_df('../data/lab3/lab3_poly2.csv', delete_first=False)
    print_df_data(data)
    xp_train, xp_test, yp_train, yp_test = split_and_show(data[["x"]], data["y"])

    linear = make_regression(xp_train, yp_train, LinearRegression(), "linear")
    show_all_metrics(xp_train, xp_test, yp_train, yp_test, linear, "linear", get_predicted_y)
    show_predicted_plot(yp_test, get_predicted_y(xp_test, linear), xp_test)

    for n in range(1, 15):
        show_poly_plot(n)
