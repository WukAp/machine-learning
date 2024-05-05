import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import tensorflow as tf
from lab1.pandas_1 import upload_df, print_df_data
from lab3.linear_regression_1 import split_and_show, show_predicted_plot, \
    make_regression, show_all_metrics, metrics


def get_predicted_poly_y(x, num):
    poly_transform = PolynomialFeatures(num)
    xnl_poly = poly_transform.fit_transform(x_train)
    lin_nl2 = LinearRegression(fit_intercept=False)
    lin_nl2.fit(xnl_poly, y_train)
    poly_line = poly_transform.transform(x.reshape(-1, 1))
    y_pred2 = lin_nl2.predict(poly_line)
    return y_pred2


def get_predicted_tensor_y(x, w):
    return w[0] * x ** 6 + w[1] * x ** 5 + w[2] * x ** 4 + w[3] * x ** 3 + w[4] * x ** 2 + w[5] * x + w[6]


def show_metrics_plot(x_train, x_test, y_train, y_test):
    N = 58
    r_train = np.zeros(N)
    r_test = np.zeros(N)
    for num in range(N):
        r_train[num] = r2_score(y_train, get_predicted_poly_y(x_train, num))
        r_test[num] = r2_score(y_test, get_predicted_poly_y(x_test, num))
        print("\n========== train r2 n=", num, " ==========")
        print(r_train[num])
        print("========== test r2 n=", num, " ==========")
        print(r_test[num])
    plt.plot(range(N), r_train, 'b-')
    plt.plot(range(N), r_test, 'r-')
    plt.grid()
    plt.legend(("train", "test"))

    plt.show()


def make_nonlinear_poly_regression(n, x_train, y_train):
    poly_transform = PolynomialFeatures(n)
    xnl_poly = poly_transform.fit_transform(x_train)
    linear = LinearRegression(fit_intercept=False)
    linear.fit(xnl_poly, y_train)
    return linear


def show_poly_plot(num):
    x_line_nl = x_train
    y = get_predicted_poly_y(x_line_nl, num)
    plt.plot(x_train, y_train, 'b.')
    plt.plot(x_test, y_test, 'r.')
    plt.plot(x_line_nl, y, 'k.')
    plt.legend(('train', 'test', 'predict'))
    plt.title("PolynomialFeatures " + str(num))
    plt.grid()
    plt.show()


def make_nonlinear_tensor_regression(Y):
    x_tf = tf.constant(np.array(x_train), dtype=tf.float32)
    y_tf = tf.constant(np.array(y_train), dtype=tf.float32)
    w = [tf.Variable(np.random.randn()) for _ in range(7)]
    alpha = tf.constant(0.2, dtype=tf.float32)
    epoch_n = 9000
    for epoch in range(epoch_n):
        with tf.GradientTape() as tape:
            y_pred = get_predicted_tensor_y(x_tf, w)
            loss = tf.reduce_mean(tf.square(y_tf - y_pred))
        grad = tape.gradient(loss, w)
        for i in range(7):
            w[i].assign_add(-(alpha * grad[i]))
        if (epoch + 1) % 750 == 0:
            print(f"E: {epoch + 1}, L: {loss.numpy()}")
    return w


def show_tensor_plot(w, x_test):
    y_line_test = get_predicted_tensor_y(x_test, w)
    plt.plot(x_train, y_train, 'k.')
    plt.plot(x_test, y_test, 'r.')
    plt.plot(x_test, y_line_test, 'b.')
    plt.grid()
    plt.title('tensor flow')
    plt.legend(("train", "test", "test_predict"))
    plt.show()


def show_plot(x_train, x_test, y_train, y_test):
    plt.scatter(x_train, y_train, c='b', marker='.')
    plt.scatter(x_test, y_test, c='r', marker='.')
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(('train', 'test'))
    plt.show()


if __name__ == '__main__':
    data = upload_df('../data/lab3/lab3_poly2.csv', delete_first=False)
    print_df_data(data)
    x_split = np.array(data["x"]).reshape(-1, 1)
    y_split = np.array(data["y"]).reshape(-1, 1)
    x_train, x_test, y_train, y_test = (
        train_test_split(x_split, y_split, test_size=0.3))

    show_plot(x_train, x_test, y_train, y_test)

    linear = make_regression(x_train, y_train, LinearRegression(), "linear")
    show_all_metrics(y_train, y_test, linear.predict(x_train), linear.predict(x_test), "linear")
    show_predicted_plot(x_test, y_test, linear.predict(x_test))
    show_metrics_plot(x_train, x_test, y_train, y_test)
    for n in range(6, 7):
        reg = make_nonlinear_poly_regression(n, x_train, y_train)
        show_all_metrics(y_train, y_test, get_predicted_poly_y(x_train, n), get_predicted_poly_y(x_test, n), "poly")
        show_poly_plot(n)
    w = make_nonlinear_tensor_regression(y_train)
    print(w)
    show_all_metrics(y_train, y_test, get_predicted_tensor_y(x_train, w), get_predicted_tensor_y(x_test, w), "poly")
    show_tensor_plot(w, x_test)
