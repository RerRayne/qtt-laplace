# coding=UTF-8

import numpy as np

from qtt_laplace.basis.integrate_basis import get_local_stiffness_integral


def stiffness_values(x_numbers, y_numbers, lx_1, ly_1, lx_2, ly_2, mesh_container, basis_class, leggauss_deg=1):
    """
    Position of a mesh finite element describes by it order from left bottom corner.
    ----*----
    | C | D |
    ----*----
    | A | B |
    ----*----

    For example:
    A has patch_x=0, patch_y=0;
    B has patch_x=1, patch_y=0
    C has patch_x=0, patch_y=1
    D has patch_x=1, patch_y=1

    Let's imagine that we map each finite element from source space to square ([-1, 1],[-1, 1]).
    So, we want to calculate an integral between two basis function (lx_1, ly_1) and (lx_2, ly_2)
    on each mapped finite element.
    This function returns a vector of integral's value on each finite element.

    :param x_numbers: array with numbers of finite elements along x-axe.
    :param y_numbers: array with numbers of finite elements along y-axe.
    :param lx_1: position of the first basis function by x-axe. Should be -1 if x=-1 and 1 if x=1;
    :param ly_1: position of the first basis function by y-axe. Should be -1 if y=-1 and 1 if y=1;
    :param lx_2: position of the second basis function by x-axe. Should be -1 if x=-1 and 1 if x=1;
    :param ly_2: position of the second basis function by y-axe. Should be -1 if y=-1 and 1 if y=1;
    :param mesh_container: this generator returns jacobian function depends on the patch_x and patch_y;
    :param basis_class: class, which describes basis function. Should be inherited from class N; # TODO: do base class
    :param leggauss_deg: Gauss-Legendre quadrature degree. By default leggauss_deg=1;
    :return: vector of integral's value on each finite element.
    """
    x_numbers = np.array(x_numbers).astype(np.int)
    y_numbers = np.array(y_numbers).astype(np.int)

    return [get_local_stiffness_integral(
                                        lx_1, ly_1, lx_2, ly_2,
                                        mesh_container.jac(x, y),
                                        basis_class,
                                        leggauss_deg) for x, y in zip(x_numbers, y_numbers)]
