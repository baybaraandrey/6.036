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

