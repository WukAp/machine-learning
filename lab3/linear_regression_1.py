from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, SGDRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from lab1.pandas_1 import upload_df, print_df_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt


def split_and_show(xp_split, yp_split, test_size=0.3, shuffle=False,  alpha=1):
    x_train, x_test, y_train, y_test = train_test_split(xp_split, yp_split, test_size=test_size, shuffle=shuffle)
    for title in xp_split.columns.tolist():
        plt.scatter(x_train[title], y_train, c='b', marker='.', alpha =alpha)
        plt.scatter(x_test[title], y_test, c='r', marker='.', alpha =alpha)
        plt.grid()
        plt.xlabel(title)
        plt.ylabel("y")
        plt.legend(('train', 'test'))
        plt.show()
    return x_train, x_test, y_train, y_test


def metrics(y_real, y_predicted):
    print("r: ", r2_score(y_real, y_predicted))
    print("MAPE: ", mean_absolute_percentage_error(y_real, y_predicted) * 100, "%")
    print("MAE: ", mean_absolute_error(y_real, y_predicted))


def make_regression(x_train, y_train, regression, regression_name):
    regression.fit(x_train, y_train)
    print("С помощью", regression_name, "получены коэффициенты при x", regression.coef_,
          "Полученный свободный член ", regression.intercept_)
    return regression


def show_all_metrics(y_train, y_test, y_train_pred, y_test_pred, regression_name):
    print("========== " + regression_name + " train metrics ==========")
    metrics(y_train, y_train_pred)
    print("========== " + regression_name + " test metrics ==========")
    metrics(y_test, y_test_pred)


def show_predicted_plot(x, y_real, y_predicted, x_lable='x', y_lable='y'):
    plt.plot(x, y_real, 'k.')
    plt.plot(x, y_predicted, 'rx')
    plt.legend(('real', 'predict'))
    plt.title("test sample prediction")
    plt.xlabel(x_lable)
    plt.ylabel(y_lable)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    data = upload_df('../data/lab3/lab3_lin3.csv', delete_first=False)
    print_df_data(data)
    x_train, x_test, y_train, y_test = split_and_show(data[["x1", "x2", "x3", "x5"]], data["y"])

    linear = make_regression(x_train, y_train, LinearRegression(), "linear")
    show_all_metrics(y_train, y_test, linear.predict(x_train), linear.predict(x_test), "linear")
    lasso = make_regression(x_train, y_train, Lasso(0.09), "lasso")
    show_all_metrics(y_train, y_test, lasso.predict(x_train), lasso.predict(x_test), "lasso")

    ridge = make_regression(x_train, y_train, Ridge(0.1), "ridge")
    show_all_metrics(y_train, y_test, ridge.predict(x_train), ridge.predict(x_test), "ridge")

    elastic = make_regression(x_train, y_train, ElasticNet(alpha=0.13), "elastic")
    show_all_metrics(y_train,y_test, elastic.predict(x_train), elastic.predict(x_test), "elastic")
    elastic = make_regression(x_train, y_train, ElasticNet(alpha=0.10), "elastic")
    show_all_metrics(y_train,y_test, elastic.predict(x_train), elastic.predict(x_test), "elastic")
    elastic = make_regression(x_train, y_train, ElasticNet(alpha=0.2), "elastic")
    show_all_metrics(y_train,y_test, elastic.predict(x_train), elastic.predict(x_test), "elastic")

    gradient = make_regression(x_train, y_train, SGDRegressor(), "gradient")
    show_all_metrics(y_train, y_test, gradient.predict(x_train), gradient.predict(x_test), "gradient")
    for title in x_test.columns.tolist():
        show_predicted_plot(x_test[title], y_test, lasso.predict(x_test), title)
