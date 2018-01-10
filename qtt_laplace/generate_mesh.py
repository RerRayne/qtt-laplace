# coding=UTF-8
import numpy as np
import tt

from qtt_laplace.basis.linear_basis.interpolate_linear import interpolate_linear
from qtt_laplace.basis.permutation_matrix import get_permutation_matrix
from qtt_laplace.utils import det


def get_jacobian_components(mesh_container):
    """
    :param mesh_container:
    :return:
    """

    d = mesh_container.d

    j11 = interpolate_linear(lambda ex, ey: mesh_container.jac0(ex, ey)[0, 0], d)
    j12 = interpolate_linear(lambda ex, ey: mesh_container.jac0(ex, ey)[0, 1], d)
    j21 = interpolate_linear(lambda ex, ey: mesh_container.jac0(ex, ey)[1, 0], d)
    j22 = interpolate_linear(lambda ex, ey: mesh_container.jac0(ex, ey)[1, 1], d)

    return j11, j12, j21, j22


def get_jacobian_determinants(mesh_container):
    """

    :param mesh_container:
    :return:
    """
    return interpolate_linear(lambda ex, ey: det(mesh_container.jac0(ex, ey)), mesh_container.d)


def get_block_indexes():
    """

    :return:
    """
    return np.array(np.meshgrid(*[range(2)] * 4, indexing='ij', copy=False)).reshape(4, -1).T


def build_Kl(basis_function, lx, ly, lxs, lys, TJ11, TJ12, TJ22, eps):
    """

    :return:
    """

    # \nabla\phi_i J^A J^AT/detJ \nabla\phi_j в точке 0, 0
    # 4 - это площадь стандартного квадрата

    grad_i = basis_function.grad_phi_2d(lx, ly, 0, 0)
    grad_j = basis_function.grad_phi_2d(lxs, lys, 0, 0)

    quad_area = 4  # TODO: вынести в более логически обоснованое место
    Kl = quad_area * (grad_i[0] * grad_j[0] * TJ11 + grad_i[1] * grad_j[1] * TJ22 + \
                      (grad_i[0] * grad_j[1] + grad_i[1] * grad_j[0]) * TJ12)
    return Kl.round(eps)


def build_Gl(basis_function, lx, ly, lxs, lys, detJ, eps):
    """

    :return:
    """
    Fi = basis_function.phi_2d(lx, ly, 0, 0)
    Fj = basis_function.phi_2d(lxs, lys, 0, 0)
    quad_area = 4
    Gl = quad_area * Fi * Fj * detJ
    return Gl.round(eps)


def build_Al(Kl, P, lx, ly, lxs, lys, eps):
    Al = P[lx, ly].T * tt.diag(Kl) * P[lxs, lys]
    return Al.round(eps)


def build_Ml(Gl, P, lx, ly, lxs, lys, eps):
    Ml = P[lx, ly].T * tt.diag(Gl) * P[lxs, lys]
    return Ml.round(eps)


def add_to_final_stiffness_matrix(A, Al, eps):
    # Впечатываем куда надо диагональные элементы
    if A is None:
        A = Al
    else:
        A += Al

    return A.round(eps)


def add_to_final_mass_matrix(f, Ml, eps, d):
    rhs = tt.ones(4, d)

    # Впечатываем куда надо диагональные элементы
    if f is None:
        f = tt.matvec(Ml, rhs)
    else:
        f += tt.matvec(Ml, rhs)

    return f.round(eps)


def get_space_transorm_jacobians(mesh_container, eps, verbose=False):
    detJ = get_jacobian_determinants(mesh_container)

    # 1/detJ
    idetJ = tt.multifuncrs([detJ], lambda x: 1. / x, eps=eps, verb=False)

    if verbose: print 'idetJ.erank =', idetJ.erank

    # J = [[j11, j12], [j21, j22]]
    j11, j12, j21, j22 = get_jacobian_components(mesh_container)

    # J^A J^AT/detJ в векторизованом виде (для каждого элемента)
    TJ11 = (j22 * j22 + j12 * j12) * idetJ
    TJ11 = TJ11.round(eps)

    TJ22 = (j11 * j11 + j21 * j21) * idetJ
    TJ22 = TJ22.round(eps)

    TJ12 = -(j22 * j21 + j12 * j11) * idetJ
    TJ12 = TJ12.round(eps)

    return TJ11, TJ12, TJ22, detJ


def assemble_on_quad(mesh_container, basis_function, eps=1e-8, verbose=False):
    """
    Au = Mf
    Assemble stiffness matrix, momentum matrix, force vector for quadrilateral in TT
    :param eps:
    :param basis_function:
    :param verbose:
    :param mesh_container:
    :return A, f in TT format with z-curve
    """
    A = None
    f = None
    P = get_permutation_matrix(mesh_container.d)
    TJ11, TJ12, TJ22, detJ = get_space_transorm_jacobians(mesh_container, eps, verbose)

    for lx, ly, lxs, lys in get_block_indexes():
        # Вычисляем градиенты в центре FE
        Kl = build_Kl(basis_function, lx, ly, lxs, lys, TJ11, TJ12, TJ22, eps)
        if verbose: print Kl.full()

        # Теперь функции для правой части{c_1 c_2}
        Gl = build_Gl(basis_function, lx, ly, lxs, lys, detJ, eps)

        if verbose: print 'lx = %d,ly = %d,lxs = %d,lys = %d' % (lx, ly, lxs, lys)
        if verbose: print 'Kl.erank =', Kl.erank, 'Gl.erank =', Gl.erank

        # Au = Mf
        Al = build_Al(Kl, P, lx, ly, lxs, lys, eps)
        Ml = build_Ml(Gl, P, lx, ly, lxs, lys, eps)
        if verbose: print 'Al.erank =', Al.erank, 'Ml.erank =', Ml.erank

        # Впечатываем куда надо диагональные элементы
        A = add_to_final_stiffness_matrix(A, Al, eps)
        f = add_to_final_mass_matrix(f, Ml, eps, mesh_container.d)

    return A, f


def get_interfaces(A, f, sides):
    """
    :param A: stiffness matrix
    :param f: force vector
    :param sides: sewing sides
    :return: modified vectors A, f and interface matrices in TT format with z-curve
    """
    pass


def solve():
    pass


def plot():
    pass
