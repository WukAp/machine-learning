from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, SGDRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

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
        plt.legend(('train', 'test'))
        plt.title(title)
        plt.show()
    return xp_train, xp_test, yp_train, yp_test


def get_predicted_y(xp, reg):
    return reg.coef_[0] * xp["x1"] + reg.coef_[1] * xp["x2"] + reg.coef_[2] * xp["x3"] + reg.coef_[3] * xp[
        "x5"] + reg.intercept_


def metrics(y_real, y_predicted):
    print("r: ", r2_score(y_real, y_predicted))
    print("MAPE: ", mean_absolute_percentage_error(y_real, y_predicted) * 100)
    print("MAE: ", mean_absolute_error(y_real, y_predicted))


def make_regression(xp_train, yp_train, regression, regression_name):
    regression.fit(xp_train, yp_train)
    print("С помощью", regression_name, "Полученные коэффициенты при x ", regression.coef_,
          "Полученный свободный член ", regression.intercept_)
    return regression


def show_all_metrics(xp_train, xp_test, yp_train, yp_test, regression, regression_name,
                     get_predicted_y=get_predicted_y):
    print("========== " + regression_name + " train metrics ==========")
    metrics(yp_train, get_predicted_y(xp_train, regression))
    print("========== " + regression_name + " test metrics ==========")
    metrics(yp_test, get_predicted_y(xp_test, regression))


def show_predicted_plot(y, y_predicted, x):
    for title in x.columns.tolist():
        plt.plot(x[title], y, 'k.')
        plt.plot(x[title], y_predicted, 'rx')
        plt.legend(('real', 'predict'))
        plt.title("test sample prediction")
        plt.grid()
        plt.show()


if __name__ == '__main__':
    data = (upload_df('../data/lab3/lab3_lin3.csv', delete_first=False))
    print_df_data(data)
    xp_train, xp_test, yp_train, yp_test = split_and_show(data[["x1", "x2", "x3", "x5"]], data["y"])

    linear = make_regression(xp_train, yp_train, LinearRegression(), "linear")
    show_all_metrics(xp_train, xp_test, yp_train, yp_test, linear, "linear", get_predicted_y)

    lasso = make_regression(xp_train, yp_train, Lasso(0.09), "lasso")
    show_all_metrics(xp_train, xp_test, yp_train, yp_test, lasso, "lasso", get_predicted_y)

    ridge = make_regression(xp_train, yp_train, Ridge(0.1), "ridge")
    show_all_metrics(xp_train, xp_test, yp_train, yp_test, ridge, "ridge", get_predicted_y)

    elastic = make_regression(xp_train, yp_train, ElasticNet(alpha=0.127, max_iter=1000), "elastic")
    show_all_metrics(xp_train, xp_test, yp_train, yp_test, elastic, "elastic", get_predicted_y)

    regressor = make_regression(xp_train, yp_train, SGDRegressor('epsilon_insensitive'), "regressor")
    show_all_metrics(xp_train, xp_test, yp_train, yp_test, regressor, "regressor", get_predicted_y)

    show_predicted_plot(yp_test, get_predicted_y(xp_test, elastic), xp_test)
