# coding=UTF-8

from __future__ import print_function

import sys


class LinearPhi1D:

    def __init__(self):
        pass

    @staticmethod
    def phi_1d(l, xi):
        """
        One-dimensional basis function on the segment [-1, 1].
        Basis function l = -1 - is the function which is equal to one in -1 end equal to 0 in point 1.
        For l = 1 vice versa.
        :param l: number of basis function;
        :param xi: desired point;
        :return the value of basis function in point xi;
        """
        LinearPhi1D.check_basis_function(l)
        l = LinearPhi1D.transform_l(l)

        m = l - 0.5  # l == 0 -> m = -0.5, l == 1 -> m = 0.5
        return max(0.5 + m * xi, 0)

    @staticmethod
    def dphi_1d(l):
        """
        The derivative of one-dimensional basis function.
        :param l: number of basis function;
        :return derivative value;
        """
        LinearPhi1D.check_basis_function(l)
        l = LinearPhi1D.transform_l(l)

        return l - 0.5

    @staticmethod
    def check_basis_function(l):
        """
        Check number of basis function.
        If something goes wrong, method will raise ValueError exception.
        :param l: number of basis function;
        """
        # TODO: вообще-то l=0, ставящее нодалку  в точку -1 контринтуитивна, надо будет заменить.
        accepted_values = [-1, 0, 1]
        if l not in accepted_values:
            raise ValueError("l should be equal to 0 or 1")

    @staticmethod
    def transform_l(l):
        """
        Little math trick with basis function number.
        :param l: basis number;
        :return transformed basis function;
        """
        if l == 0:
            # print("WARNING! Deprecated behaviour. Replace 0 to -1", file=sys.stderr)
            return l
        elif l == -1:
            return 0
        else:
            return l

    @staticmethod
    def check_basis_function(l):
        """
        Check number of basis function.
        If something goes wrong, method will raise ValueError exception.
        :param l: number of basis function;
        """
        # TODO: вообще-то l=0, ставящее нодалку  в точку -1 контринтуитивна, надо будет заменить.
        accepted_values = [-1, 0, 1]
        if l not in accepted_values:
            raise ValueError("l should be equal to 0 or 1")
