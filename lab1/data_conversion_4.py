import numpy as np
from pandas_1 import *
from numpy_2 import *
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler


def print_encoders(df_labels):
    le = LabelEncoder()
    le.fit(df_labels)
    print("LabelEncoder transformation: ", le.classes_, " -> ", le.transform(le.classes_))

    ohe = OneHotEncoder(sparse_output=False)
    ohe.fit(np.array(df_labels).reshape(-1, 1))
    print("OneHotEncoder transformation: ", ohe.categories_, " ->\n",
          ohe.transform(np.array(ohe.categories_[0].reshape(-1, 1))))


def show_hist(a, name, axs, n, columns_names):
    for i in range(4):
        axs[n][i].hist(a[:, i], bins=15, fill=False, ec='purple')
        axs[n][i].set_title(name)
        axs[n][i].set_xlabel(columns_names[i])


def show_scaler_sklearn(scaler, scaler_name, a_features, axs, n, columns_names):
    scaler.fit(a_features)
    a = scaler.transform(a_features)
    show_hist(a, scaler_name, axs, n, columns_names)


def show_scalers_sklearn(a_features, columns_names):
    ig, axs = plt.subplots(4, 4, figsize=(18, 10))

    show_scaler_sklearn(StandardScaler(), "StandardScaler", a_features, axs, 0, columns_names)
    show_scaler_sklearn(MinMaxScaler(), "MinMaxScaler", a_features, axs, 1, columns_names)
    show_scaler_sklearn(MaxAbsScaler(), "MaxAbsScaler", a_features, axs, 2, columns_names)
    show_scaler_sklearn(RobustScaler(), "RobustScaler", a_features, axs, 3, columns_names)
    plt.tight_layout()
    plt.show()


def get_my_std_scaled_a(a_features):
    ex = np.mean(a_features, axis=0)
    qx = np.std(a_features, axis=0)
    return (a_features - ex) / qx


def get_my_min_max_scaled_a(a_features):
    min_x = np.min(a_features, axis=0)
    max_x = np.max(a_features, axis=0)
    return (a_features - min_x) / (max_x - min_x)


def print_metrics(name, a):
    print(name, " min: ", np.min(a, axis=0))
    print(name, " max: ", np.max(a, axis=0))
    print(name, "mean: ", np.mean(a, axis=0))
    print(name, "std: ", np.std(a, axis=0), "\n")


def show_my_scalers(a_features, columns_names):
    ig, axs = plt.subplots(4, 4, figsize=(18, 10))

    a = StandardScaler().fit(a_features).transform(a_features)
    print_metrics("StandardScaler, sklearn", a)
    show_hist(a, "StandardScaler, sklearn", axs, 0, columns_names)

    a = MinMaxScaler().fit(a_features).transform(a_features)
    print_metrics("MinMaxScaler, sklearn", a)
    show_hist(a, "MinMaxScaler, sklearn", axs, 1, columns_names)

    a = get_my_std_scaled_a(a_features)
    print_metrics("My StandardScaler", a)
    show_hist(a, "My StandardScaler", axs, 2, columns_names)

    a = get_my_min_max_scaled_a(a_features)
    print_metrics("My MinMaxScaler", a)
    show_hist(a, "My MinMaxScaler", axs, 3, columns_names)
    plt.tight_layout()
    plt.show()


def get_filtered_df(df):
    p_25 = df['sepal width (cm)'].quantile(0.25)
    p_75 = df['sepal width (cm)'].quantile(0.75)
    print(p_25, p_75)
    df = df[(df['target'].isin(['Iris-versicolor', 'Iris-virginica'])) & (p_25 <= df['sepal width (cm)']) & (
                df['sepal width (cm)'] <= p_75)]
    return df[['sepal length (cm)', 'petal length (cm)', 'target']]


columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
if __name__ == '__main__':
    iris_df = upload_df('../data/lab1/replaced_target_iris.csv', delete_first=False)
    iris_df_labels = iris_df['target']

    print_encoders(iris_df_labels);

    iris_a_features = upload_df('../data/lab1/replaced_target_iris.csv', delete_first=False).iloc[:, 0:4].to_numpy()
    show_scalers_sklearn(iris_a_features, columns)
    show_my_scalers(iris_a_features, columns)
    print(get_filtered_df(iris_df))
