# coding=UTF-8
import itertools
import numpy as np


def interpolate_func(p1, p2, p3, p4, x, y):
    return p1 * (1 - x) * (1 - y) + p2 * x * (1 - y) + p3 * x * y + p4 * (1 - x) * y


class QuadMeshContainer(object):
    def _pts(self, i, j):
        n = self.n
        p1 = self.p1
        p2 = self.p2
        p3 = self.p3
        p4 = self.p4

        x = i / (n - 1.)  # точки на сетке с равномерным шагом
        y = j / (n - 1.)  # точки на сетке с равномерным шагом
        return p1 * (1 - x) * (1 - y) + p2 * x * (1 - y) + p3 * x * y + p4 * (1 - x) * y

    def __init__(self, p1, p2, p3, p4, d):
        """

        :param p1:
        :param p2:
        :param p3:
        :param p4:
        :param d:
        """

        # self.d = d
        # self.n = 2 ** d
        # p1 = np.array(p1, dtype=np.double)
        # p2 = np.array(p2, dtype=np.double)
        # p3 = np.array(p3, dtype=np.double)
        # p4 = np.array(p4, dtype=np.double)
        #
        # points_number = self.n + 1
        # norm = self.n - 1
        # s = np.arange(0, points_number, dtype=np.double) / norm
        #
        # self.pts = np.array([interpolate_func(p1, p2, p3, p4, i, j) for i, j in
        #                      itertools.product(s, s)]).reshape((points_number, points_number, 2))

        n = 2 ** d
        p1 = np.array(p1, dtype=np.double)
        p2 = np.array(p2, dtype=np.double)
        p3 = np.array(p3, dtype=np.double)
        p4 = np.array(p4, dtype=np.double)

        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4

        self.d = d
        self.n = n

    def jac(self, x_number, y_number):
        """

        :param x_number:
        :param y_number:
        :return:
        """

        x1, y1 = self._pts(x_number, y_number)
        x2, y2 = self._pts(x_number+1, y_number)
        x3, y3 = self._pts(x_number+1, y_number+1)
        x4, y4 = self._pts(x_number, y_number+1)

        J0 = 0.25*np.array([[x2+x3-x1-x4, x3+x4-x1-x2], [y2+y3-y1-y4, y3+y4-y1-y2]])
        Jx = 0.25*np.array([[0, x1-x2+x3-x4], [0, y1-y2+y3-y4]])
        Jy = 0.25*np.array([[x1-x2+x3-x4, 0], [y1-y2+y3-y4, 0]])

        return lambda xi, eta: J0 + xi * Jx + eta * Jy

    # Jacobian at the center
    def jac0(self, ex, ey):
        """

        :param ex:
        :param ey:
        :return:
        """

        x1, y1 = self._pts(ex, ey)
        x2, y2 = self._pts(ex + 1, ey)
        x3, y3 = self._pts(ex + 1, ey + 1)
        x4, y4 = self._pts(ex, ey + 1)
        return 0.25 * np.array([
                                    [x2 + x3 - x1 - x4, x3 + x4 - x1 - x2],
                                    [y2 + y3 - y1 - y4, y3 + y4 - y1 - y2]
                                ])
