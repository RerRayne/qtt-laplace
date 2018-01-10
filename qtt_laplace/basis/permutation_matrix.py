# coding=UTF-8
import itertools
import numpy as np
import tt

from qtt_laplace.utils import tt_build_from_list
from qtt_laplace.zoperations import zkron


def get_permutation_matrix(d):
    """

    :param d:
    :return:
    """
    W = [getW0(d), getW1(d)]
    P = np.empty((2, 2), dtype=object)
    for lx, ly in itertools.product(range(2), range(2)):
        P[lx, ly] = zkron(W[lx], W[ly])

    return P


class W0Methods:
    @classmethod
    def getFirstCore(self):
        # Еще одна breaking news.
        # Индексы в блочной матрице ядра воспринимаются tt следующем порядке:
        # (номер столбца в блочной матрице; номер строчки в блоке; номер столбца в блоке; номер строчки блока)
        core = np.zeros((2, 2, 2, 1))
        core[0, :, :, 0] = [[0, 0], [0, 1]]
        core[1, :, :, 0] = [[1, 0], [0, 0]]
        return core

    @classmethod
    def getMiddleCore(self):
        core = np.zeros((2, 2, 2, 2))
        core[0, :, :, 0] = [[0, 0], [0, 1]]
        core[1, :, :, 0] = [[1, 0], [0, 0]]
        core[0, :, :, 1] = [[0, 0], [0, 0]]
        core[1, :, :, 1] = [[1, 0], [0, 1]]

        return core

    @classmethod
    def getLastCore(self):
        core = np.zeros((1, 2, 2, 2))
        core[0, :, :, 0] = [[1, 0], [0, 0]]
        core[0, :, :, 1] = [[1, 0], [0, 1]]

        return core

    @classmethod
    def getForD1(self):
        return tt.matrix(np.array([[1, 0], [0, 0]]))


class W1Methods:
    @classmethod
    def getFirstCore(self):
        core = np.zeros((2, 2, 2, 1))
        core[0, :, :, 0] = [[1, 0], [0, 1]]
        core[1, :, :, 0] = [[0, 1], [0, 0]]
        return core

    @classmethod
    def getMiddleCore(self):
        core = np.zeros((2, 2, 2, 2))
        core[0, :, :, 0] = [[1, 0], [0, 1]]
        core[1, :, :, 0] = [[0, 1], [0, 0]]
        core[0, :, :, 1] = [[0, 0], [0, 0]]
        core[1, :, :, 1] = [[0, 0], [1, 0]]

        return core

    @classmethod
    def getLastCore(self):
        core = np.zeros((1, 2, 2, 2))
        core[0, :, :, 0] = [[0, 1], [0, 0]]
        core[0, :, :, 1] = [[0, 0], [1, 0]]

        return core

    @classmethod
    def getForD1(self):
        return tt.matrix(np.array([[0, 1], [0, 0]]))


def constructW(d, W):
    if d < 1:
        raise ValueError("Wrong d-parameter. d should be greater or equal 1.")

    if d == 1:
        return W.getForD1()

    cores = [W.getFirstCore()]

    if d > 2:
        cores += [W.getMiddleCore()] * (d - 2)

    cores.append(W.getLastCore())

    return tt_build_from_list(cores)


def getW0(d):
    return constructW(d, W0Methods)


def getW1(d):
    return constructW(d, W1Methods)