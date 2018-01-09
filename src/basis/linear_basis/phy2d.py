# coding=UTF-8
import numpy as np

from ttlaplace.basis.linear_basis.phi1d import LinearPhi1D


class LinearPhi2D:

    def __init__(self):
        pass

    @staticmethod
    def grad_phi_2d(lx, ly, xi, eta):
        """
        Gradient of two-dimensional basis function on the segment [(-1, 1), (-1, 1)].
        :param lx: number of basis function by x-axe. Should be 0 if x=-1 and 1 if x=1;
        :param ly: number of basis function by y-axe. Should be 0 if y=-1 and 1 if y=1;
        :param xi: x-coordinate of desired point;
        :param eta: y-coordinate of desired point;
        :return gradient vector of basis function in point (xi, eta);
        """
        return np.array([LinearPhi1D.dphi_1d(lx) * LinearPhi1D.phi_1d(ly, eta),
                         LinearPhi1D.phi_1d(lx, xi) * LinearPhi1D.dphi_1d(ly)])

    @staticmethod
    def phi_2d(lx, ly, xi, eta):
        """
        Two-dimensional basis function on the segment [(-1, 1), (-1, 1)].
         lx=-1, ly=1     lx=1, ly=1
                *---1---*
                |   |   |
             -1 ----0---+ 1
                |   |   |
                *---1---*
        lx=-1, ly=0      lx=1, ly=-1

        :param lx: number of basis function by x-axe. Should be -1 if x=-1 and 1 if x=1;
        :param ly: number of basis function by y-axe. Should be -1 if y=-1 and 1 if y=1;
        :param xi: x-coordinate of desired point;
        :param eta: y-coordinate of desired point;
        :return the value of basis function, defined by lx, ly in point xi, eta;
        """
        LinearPhi2D.check_two_dimensional_basis_function(lx, ly)

        return max(LinearPhi1D.phi_1d(lx, xi) * LinearPhi1D.phi_1d(ly, eta), 0)

    @staticmethod
    def check_two_dimensional_basis_function(lx, ly):
        """
        Check number of two-dimensional basis function.
        If something goes wrong, method will raise ValueError exception.
        :param lx: x-number of two-dimensional basis function;
        :param ly: y-number of two-dimensional basis function;
        """
        LinearPhi1D.check_basis_function(lx)
        LinearPhi1D.check_basis_function(ly)
