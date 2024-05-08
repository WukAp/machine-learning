from matplotlib import pyplot as plt

from lab3.linear_regression_1 import split_and_show
from lab4.kNN_2 import show_decision_boundary, show_confusion_matrix, show_metrix
from lab4.upload_1 import get_split_data

from sklearn.linear_model import LogisticRegression


def logistic_regression(x, y, penalty=None, solver='lbfgs'):
    logistic_regression = LogisticRegression(penalty=penalty, solver=solver)
    logistic_regression.fit(x, y.to_numpy())
    return logistic_regression


def show_accurancy(x_train, x_test, y_train, y_test):
    neighbors = ['None','l1',  'l2' ]
    accuracy = []
    log_none = LogisticRegression(None)
    log_none.fit(x_train, y_train)
    accuracy.append(log_none.score(x_test, y_test))
    log_l1 = LogisticRegression('l1', solver='liblinear')
    log_l1.fit(x_train, y_train)
    accuracy.append(log_l1.score(x_test, y_test))
    log_l2 = LogisticRegression('l2')
    log_l2.fit(x_train, y_train)
    accuracy.append(log_l2.score(x_test, y_test))

    print(accuracy)
    plt.bar(neighbors, accuracy)
    plt.title('Зависимость точности от регуляризации')
    plt.xlabel('Регуляризация')
    plt.ylabel('Точность')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    x, y = get_split_data()
    x_train, x_test, y_train, y_test = split_and_show(x, y, alpha=0.2)
    show_accurancy(x_train, x_test, y_train, y_test)
    logistic_regression = logistic_regression(x_train, y_train)
    show_decision_boundary(x.to_numpy(), y.to_numpy(), logistic_regression)
    show_confusion_matrix(x_test, y_test, logistic_regression)
    show_metrix(x_test, y_test, logistic_regression)
