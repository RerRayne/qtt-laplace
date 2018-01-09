# coding=UTF-8

import numpy as np

import tt


def adj(A):
    """
    Return an adjusted matrix of matrix J with size 2x2
    :param A: matrix 2x2
    :return: adjusted matrix of matrix J
    """
    [a, b], [c, d] = A
    return np.array([[d, -b], [-c, a]])


def det(A):
    """
    Return det of 2x2 matrix A.
    We implemented our own function because standard numpy function is too slow for us.
    :param A: matrix 2x2
    :return: det(A)
    """
    [a, b], [c, d] = A
    return a*d - b*c


def tt_build_from_list(cores_list):
    """
    This function takes list of cores and return TT-matrix with this cores.
    :param cores_list: - list of cores;
    :return: TT matrix.
    """
    return tt.matrix.from_list(list(reversed(cores_list)))


# Второй пункт надоело
def tt_reshape(matrix, shape):
    """
    This function reshape matrix in order="F". order="F" because TT library works fith fortran's order.
    :param matrix: numpy matrix;
    :param shape: desired shapes;
    :return: reshaped matrix.
    """
    return matrix.reshape(shape, order='F')
