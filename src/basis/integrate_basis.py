# coding=UTF-8
from __future__ import print_function

import itertools

import ttlaplace.utils as tt_utils

from numpy.polynomial.legendre import leggauss


def get_local_stiffness_integral(lx_1, ly_1, lx_2, ly_2, jacobian_func, basis_class, leggauss_deg=1):
    """
    In local stiffness matrix we should calculate an integral between
    two gradient of basis functions on a finite element.
    Each source finite element mesh transform to quad [[-1, 1], [-1, 1]]. We do integration on it.
    That's why integral transform to:
    \int_{-1}^1 \nabla \phi_1 \J^A \nabla \phi_2 \J^A/det(J)
    Eventually, we calculate integral between two quarters of gradient basis function.
    Because only quarter can locate on one finite element.
    Coordinate of 'center' of basis function describes by two value lx and ly.
    Both are equal to -1 or 1 in case of linear basis.
    lx=-1, ly=1     lx=1, ly=1
            *---1---*
            |   |   |
         -1 ----0---+ 1
            |   |   |
            *---1---*
    lx=-1, ly=0      lx=1, ly=-1

    For more details about Gauss-Legendre quadrature look at:
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.polynomial.legendre.leggauss.html
    :param lx_1: position of the first basis function by x-axe. Should be -1 if x=-1 and 1 if x=1;
    :param ly_1: position of the first basis function by y-axe. Should be -1 if y=-1 and 1 if y=1;
    :param lx_2: position of the second basis function by x-axe. Should be -1 if x=-1 and 1 if x=1;
    :param ly_2: position of the second basis function by y-axe. Should be -1 if y=-1 and 1 if y=1;
    :param jacobian_func: function, which return Jacobian matrix matrix in the particular point (xi, eta);
    :param basis_class: basis function class;
    :param  leggauss_deg: Gauss-Legendre quadrature degree. By default leggauss_deg=1;
    :return integral value between bases functions (lx_1, ly_1), (lx_2, ly_2) one one finite element;
    """
    basis_class.check_two_dimensional_basis_function(lx_1, ly_1)
    basis_class.check_two_dimensional_basis_function(lx_2, ly_2)

    return sum(
        [get_bases_value_in_point(lx_1, ly_1, lx_2, ly_2, xi, eta, wx, wy, jacobian_func, basis_class)
         for (xi, wx), (eta, wy) in get_integration_mesh(leggauss_deg)]
    )


def get_local_mass_integral(lx_1, ly_1, lx_2, ly_2, jacobian_func, basis_class, leggauss_deg=1):
    """
    One element of local mass function.
    Each source finite element mesh transform to quad [[-1, 1], [-1, 1]]. We do integration on it.
    That's why integral transform to:
    \int_{-1}^1 \phi_1 \phi_2 det(J)
     Eventually, we calculate integral between two quarters of gradient basis function.
    Because only quarter can locate on one finite element.
    Coordinate of 'center' of basis function describes by two value lx and ly.
    Both are equal to -1 or 1 in case of linear basis.
    lx=-1, ly=1     lx=1, ly=1
            *---1---*
            |   |   |
         -1 ----0---+ 1
            |   |   |
            *---1---*
    lx=-1, ly=0      lx=1, ly=-1

    For more details about Gauss-Legendre quadrature look at:
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.polynomial.legendre.leggauss.html
    :param lx_1: position of the first basis function by x-axe. Should be -1 if x=-1 and 1 if x=1;
    :param ly_1: position of the first basis function by y-axe. Should be -1 if y=-1 and 1 if y=1;
    :param lx_2: position of the second basis function by x-axe. Should be -1 if x=-1 and 1 if x=1;
    :param ly_2: position of the second basis function by y-axe. Should be -1 if y=-1 and 1 if y=1;
    :param jacobian_func: function, which return Jacobian matrix matrix in the particular point (xi, eta);
    :param basis_class:  basis function class;
    :param leggauss_deg: Gauss-Legendre quadrature degree. By default leggauss_deg=1;
    :return one element of local mass function;
    """
    return sum(
        [wx * wy * basis_class.phi_2d(lx_1, ly_1, xi, eta) *\
         basis_class.phi_2d(lx_2, ly_2, xi, eta) * tt_utils.det(jacobian_func(xi, eta))
         for xi, wx, eta, wy in get_integration_mesh(leggauss_deg)]
    )


def get_integration_mesh(leggauss_deg):
    """
    Return sequences for Gauss-Legendre integration on quad [[-1, 1], [-1, 1]].
    :param leggauss_deg: degree of Gauss-Legendre quadrature;
    :return iteration via list [(x_point, x_weight, y_point,  y_weight) ... (x_point, x_weight, y_point,  y_weight)];
    """
    zeta, omega = leggauss(leggauss_deg)
    return itertools.product(zip(zeta, omega), zip(zeta, omega))


def get_bases_value_in_point(lx_1, ly_1, lx_2, ly_2, xi, eta, wx, wy, jacobian_func, basis_class):
    """
    Calculate (\nabla\phi_1 * adj_J * \nabla\phi_2 * adj_J)/det_J in point xi, eta with Gauss-Legendre weights.
    :param lx_1: position of the first basis function by x-axe. Should be -1 if x=-1 and 1 if x=1;
    :param ly_1: position of the first basis function by y-axe. Should be -1 if y=-1 and 1 if y=1;
    :param lx_2: position of the second basis function by x-axe. Should be -1 if x=-1 and 1 if x=1;
    :param ly_2: position of the second basis function by y-axe. Should be -1 if y=-1 and 1 if y=1;
    :param xi: desired x-coordinate;
    :param eta: desired y-coordinate;
    :param wx: x-dimension Gauss-Legendre weight;
    :param wy: y-dimension Gauss-Legendre weight;
    :param jacobian_func: function, which return Jacobian matrix matrix in the particular point (xi, eta);
    :param basis_class: basis function class;
    :return  value in point (xi, eta);
    """
    J = jacobian_func(xi, eta)
    det_J = tt_utils.det(J)
    adj_J = tt_utils.adj(J)

    grad_1 = basis_class.grad_phi_2d(lx_1, ly_1, xi, eta).dot(adj_J)
    grad_2 = basis_class.grad_phi_2d(lx_2, ly_2, xi, eta).dot(adj_J)

    return wx * wy * grad_1.dot(grad_2) / det_J
