# coding=UTF-8

import numpy as np

from ttlaplace.basis.integrate_basis import get_local_mass_integral


def mass_values(x_numbers, y_numbers, lx_1, ly_1, lx_2, ly_2, mesh_container, basis_class, leggauss_deg=1):
    """

    :param x_numbers:
    :param y_numbers:
    :param lx_1:
    :param ly_1:
    :param lx_2:
    :param ly_2:
    :param mesh_container:
    :param basis_class:
    :param leggauss_deg:
    :return:
    """
    x_numbers = np.array(x_numbers).astype(np.int)
    y_numbers = np.array(y_numbers).astype(np.int)

    return [get_local_mass_integral(
                                    lx_1, ly_1, lx_2, ly_2,
                                    mesh_container.jac(x, y), basis_class, leggauss_deg
                                    ) for x, y in zip(x_numbers, y_numbers)]
