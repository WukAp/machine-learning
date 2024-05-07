from matplotlib import pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from lab1.pandas_1 import upload_df, replace_df_column
from lab3.linear_regression_1 import split_and_show
from lab4.upload_1 import get_split_data, show_scatterplot
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def kNN(x, y, n_neighbors=3, weights='uniform'):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    knn.fit(x, y.to_numpy())
    return knn


def show_accurancy(x_train, x_test, y_train, y_test):
    # Определение количества соседей
    neighbors = list(range(1, 30))

    # Cоздание пустых списков для точностей обоих методов
    accuracy_unweighted = []
    accuracy_weighted = []

    # Перебор по количеству соседей
    for k in neighbors:
        # Классификатор для не взвешенного метода k-NN
        knn_unweighted = KNeighborsClassifier(n_neighbors=k, weights='uniform')
        knn_unweighted.fit(x_train, y_train)
        accuracy_unweighted.append(knn_unweighted.score(x_test, y_test))

        # Классификатор для взвешенного метода k-NN
        knn_weighted = KNeighborsClassifier(n_neighbors=k, weights='distance')
        knn_weighted.fit(x_train, y_train)
        accuracy_weighted.append(knn_weighted.score(x_test, y_test))

    # Построение графиков
    plt.plot(neighbors, accuracy_unweighted, label='Unweighted')
    plt.plot(neighbors, accuracy_weighted, label='Weighted')
    plt.title('Зависимость точности от количества соседей')
    plt.xlabel('Количество соседей')
    plt.ylabel('Точность')
    plt.legend()
    plt.show()


def show_decision_boundary(x, y, model):
    display = DecisionBoundaryDisplay.from_estimator(
        model, x, response_method='predict',
        xlabel='x1', ylabel='x2',
        grid_resolution=200, alpha=0.7)
    display.ax_.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k')
    plt.show()


def show_predicted_plot(model):
    y_pred = model.predict(x_test)
    x_y_pred = x_test
    x_y_pred['Class'] = y_pred
    show_scatterplot(x_y_pred)
    print(y_pred)


def show_confusion_matrix(x, y, model):
    ConfusionMatrixDisplay.from_estimator(model, x, y, cmap='YlGn')
    plt.show()

def show_metrix(x, y, model):
    y_pred = model.predict(x)
    print("precision_score", precision_score(y,y_pred, average='binary'))
    print("recall_score", recall_score(y,y_pred, average='binary'))
    print("f1_score", f1_score(y,y_pred, average='binary'))


if __name__ == '__main__':
    x, y = get_split_data()
    x_train, x_test, y_train, y_test = split_and_show(x, y, alpha=0.2)
    show_accurancy(x_train, x_test, y_train, y_test)
    knn = kNN(x_train, y_train, 10, 'distance')
    show_decision_boundary(x.to_numpy(), y.to_numpy(), knn)
    show_confusion_matrix(x_test, y_test, knn)
    show_metrix(x_test, y_test, knn)
