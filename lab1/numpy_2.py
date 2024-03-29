import numpy as np


def get_a(file_name, delete_first=True, delete_last=False):
    a = np.loadtxt(open(file_name, "rb"), delimiter=",", skiprows=1)
    if delete_first:
        a = np.delete(a, 0, 1)
    if delete_last:
        a = np.delete(a, a.shape[1] - 1, 1)

    return a


def print_a_head(a, n=10):
    print("\n============== head ==============")
    print(iris_a[:n])


def print_a_description(a):
    print("\n============= describe ===============")

    print('count: ', np.full(a.shape[1], a.shape[0]))
    print('mean: ', a.mean(axis=0))
    print('std: ', a.std(axis=0))
    print('min: ', a.min(axis=0))
    print('25%: ', np.percentile(a, 25, axis=0))
    print('50%: ', np.percentile(a, 50, axis=0))
    print('75%: ', np.percentile(a, 75, axis=0))
    print('max: ', a.max(axis=0))


if __name__ == '__main__':
    iris_a = get_a('../data/iris.csv')
    print_a_head(iris_a)
    print_a_description(iris_a)
