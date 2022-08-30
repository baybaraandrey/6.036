from typing import Any

import numpy as np



def signed_dist(x: np.array, th: np.array, th_0: np.array) -> np.array:
    """Returns signed distance to the X."""
    return 1 / np.linalg.norm(th) * (np.dot(th.T, x) + th_0)


def positive(x: np.array, th: np.array, th0: np.array) -> np.array:
    """Returns side of data points to the hyperplane."""
    d = signed_dist(x, th, th0)
    return np.sign(d)


def rv(value_list: list[Any]) -> np.array:
    """Returns column vector from list of values n*1."""
    return np.array([value_list]).T


def cv(value_list: list[Any]) -> np.array:
    """Returns row vector from list of values 1*n."""
    return np.array([value_list])


def positive1(data, th, th0):
    return np.sign(th.T.dot(data) + th0)

def score1(
    data: np.array,
    labels: np.array, 
    th: np.array,
    th0: np.array,
) -> np.array:
    """Returns score."""
    return np.sum(positive1(data, th, th0) == labels, axis=1)


def score(
    data: np.array,
    labels: np.array, 
    ths: np.array,
    th0s: np.array,
) -> np.array:
    """Returns score."""
    r = positive(data, ths, th0s) == labels
    i = np.argmax(np.sum(r.T, axis=0))
    return positive(data, ths, th0s) == labels



def best_separator(
    data: np.array,
    labels: np.array,
    ths: np.array,
    th0s: np.array,
)-> np.array:
    """Return first best estimator from ths, th0s candidates."""
    sr = score(data, labels, ths, th0s.T)
    i = np.argmax(np.sum(sr.T, axis=0))
    return ths[:, i:i+1], np.array([th0s[i]])


def perceptron(data, labels, params={}, hook=None):
    """
    Perceptron learning algorithm.
    returns tuple of th and th0.
    """
    T = params.get('T', 100)
    d, n = data.shape[0], data.shape[1]  
    th, th0 = np.zeros((d, 1)), np.zeros((1, 1))
    labels = labels.reshape((n, ))
    for t in range(T):
        for i in range(n):
            x = data[:, i:i+1]
            y = labels[i]
            if y*np.sum(th.T.dot(x) + th0) <= 0:
                th = th + x*y
                th0 = th0 + y
    return th, th0


def perceptron_through_the_origin(data, labels, params={}, hook=None):
    """
    Perceptron learning algorithm.
    returns th.
    """
    T = params.get('T', 100)
    d, n = data.shape[0], data.shape[1]  
    th = np.zeros((d, 1))
    labels = labels.reshape((n, ))
    for t in range(T):
        for i in range(n):
            x = data[:, i:i+1]
            y = labels[i]
            if y*np.sum(th.T.dot(x) + th0) <= 0:
                th = th + x*y
    return th


def averaged_perceptron(data, labels, params={}, hook=None):
    """has better stability..."""
    T = params.get('T', 100)
    d, n = data.shape[0], data.shape[1]  
    th, th0 = np.zeros((d, 1)), np.zeros((1, 1))
    ths, th0s = np.zeros((d, 1)), np.zeros((1, 1))
    labels = labels.reshape((n, ))
    for t in range(T):
        for i in range(n):
            x = data[:, i:i+1]
            y = labels[i]
            if y*np.sum(th.T.dot(x) + th0) <= 0:
                th = th + x*y
                th0 = th0 + y
            ths += th
            th0s += th0
    nT = n * T
    return ths/nT, th0s/nT



def eval_classifier(learner, data_train, labels_train, data_test, labels_test):
    th, th0 = learner(data_train, labels_train)
    n = data_test.shape[1]
    return score(data_test, labels_test, th, th0)/n



def eval_learning_alg(learner, data_gen, n_train, n_test, it):
    acc = 0
    for _ in range(it):
        data_train, labels_train = data_gen(n_train)
        data_test, labels_test = data_gen(n_test)
        acc += eval_classifier(learner,
            data_train, labels_train,
            data_test, labels_test,
        )
    return acc/it
