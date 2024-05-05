from sklearn.linear_model import Lasso, LinearRegression, Ridge, ElasticNet, SGDRegressor
from sklearn.model_selection import train_test_split

from lab1.pandas_1 import upload_df, replace_df_column, print_df_data
from lab3.linear_regression_1 import make_regression, show_all_metrics, show_predicted_plot, split_and_show

class_mapping = {
    'No': 0,
    'Yes': 1,
}

def get_predicted_y(xp, reg):
    return reg.coef_[0] * xp["Hours Studied"] + reg.coef_[1] * xp["Previous Scores"] + reg.coef_[2] * xp[
        "Extracurricular Activities"] + reg.coef_[3] * xp[
        "Sleep Hours"] + reg.coef_[4] * xp["Sample Question Papers Practiced"] + reg.intercept_


def prepearing_df(data):
    result_data = replace_df_column(data, 'Extracurricular Activities', class_mapping)
    result_data = result_data.dropna()
    result_data = result_data.drop_duplicates()
    return result_data


if __name__ == '__main__':
    data = upload_df('../data/lab3/Student_Performance.csv', delete_first=False)
    print_df_data(data)
    data = prepearing_df(data)
    x_train, x_test, y_train, y_test = (
        split_and_show(data[["Hours Studied", "Previous Scores", "Extracurricular Activities", "Sleep Hours",
                               "Sample Question Papers Practiced"]], data["Performance Index"], test_size=0.3, alpha=0.1))
    linear = make_regression(x_train, y_train, LinearRegression(), "linear")
    show_all_metrics(y_train, y_test, linear.predict(x_train), linear.predict(x_test), "linear")

    lasso = make_regression(x_train, y_train, Lasso(0.01), "lasso")
    show_all_metrics(y_train, y_test, lasso.predict(x_train), lasso.predict(x_test), "lasso")

    ridge = make_regression(x_train, y_train, Ridge(0.01), "ridge")
    show_all_metrics(y_train, y_test, ridge.predict(x_train), ridge.predict(x_test), "ridge")

    elastic = make_regression(x_train, y_train, ElasticNet(0.01), "elastic")
    show_all_metrics(y_train, y_test, elastic.predict(x_train), elastic.predict(x_test), "elastic")

    gradient = make_regression(x_train, y_train, SGDRegressor(), "gradient")
    show_all_metrics(y_train, y_test, gradient.predict(x_train), gradient.predict(x_test), "gradient")
    for title in x_test.columns.tolist():
        show_predicted_plot(x_test[title], y_test, get_predicted_y(x_test, elastic), title, "Performance Index")
