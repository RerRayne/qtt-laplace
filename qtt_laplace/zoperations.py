# coding=UTF-8
import numpy as np
import tt


def zkron(ttA, ttB):
    """
    Do kronecker product between two matrices ttA and ttB.
    Look about kronecker at: https://en.wikipedia.org/wiki/Kronecker_product
    :param ttA: first TT-matrix;
    :param ttB: second TT-matrix;
    :return: operation result in z-order
    """
    # TODO: ввернуть описание про работу этой функции в теор.плане или ссылку на описание
    Al = tt.matrix.to_list(ttA)
    Bl = tt.matrix.to_list(ttB)
    Hl = [np.kron(B, A) for (A, B) in zip(Al, Bl)]
    return tt.matrix.from_list(Hl)


def zkronv(ttA, ttB):
    """
    Do kronecker product between vectors ttA and ttB.
    :param ttA: first TT-vector;
    :param ttB: second TT-vector;
    :return: operation result in z-order
    """
    Al = tt.vector.to_list(ttA)
    Bl = tt.vector.to_list(ttB)
    Hl = [np.kron(B, A) for (A, B) in zip(Al, Bl)]
    return tt.vector.from_list(Hl)


def zmeshgrid(d):
    """

    :param d:
    :return:
    """
    # assert d > 2

    lin = tt.xfun(2, d)
    one = tt.ones(2, d)

    xx = zkronv(lin, one)
    yy = zkronv(one, lin)

    return xx, yy


def zaffine(c0, c1, c2, d):
    """
    Generate linear function c0 + c1 ex + c2 ey in z ordering of dimension d
    :param c0:
    :param c1:
    :param c2:
    :param d:
    :return:
    """

    xx, yy = zmeshgrid(d)
    Hx, Hy = tt.vector.to_list(xx), tt.vector.to_list(yy)

    import copy
    Hs = copy.deepcopy(Hx)
    Hs[0][:, :, 0] = c1 * Hx[0][:, :, 0] + c2 * Hy[0][:, :, 0]
    Hs[-1][1, :, :] = c1 * Hx[-1][1, :, :] + (c0 + c2 * Hy[-1][1, :, :])

    d = len(Hs)
    for k in range(1, d-1):
        Hs[k][1, :, 0] = c1 * Hx[k][1, :, 0] + c2 * Hy[k][1, :, 0]

    return tt.vector.from_list(Hs)
