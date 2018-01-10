import tt

from qtt_laplace.zoperations import zkronv


def gen_mask(xbc, ybc, d):
    """
    Generate boundary mask
    :param xbc:
    :param ybc:
    :param d:
    :return: mask
    """
    xmask = tt.ones(2, d) # O(d)
    ymask = tt.ones(2, d) # O(d)
    if xbc[0] == 'D':
        xmask = xmask - tt.unit(2, d, j=0) # O(d)
    if xbc[1] == 'D':
        xmask = xmask - tt.unit(2, d, j=2**d-1) # O(d)
    if ybc[0] == 'D':
        ymask = ymask - tt.unit(2, d, j=0) # O(d)
    if ybc[1] == 'D':
        ymask = ymask - tt.unit(2, d, j=2**d-1) # O(d)
    mask = zkronv(xmask, ymask) # O(d),because TT-ranks=1
    return tt.diag(mask) # O(d)


def apply_mask(A, f, mask, eps=1e-8):
    """
    Apply boundary mask on tt-stiffness matrix and tt-force vector
    :param A:
    :param f:
    :param mask:
    :return:
    """
    d = A.tt.d
    return (mask * A + tt.eye(4, d) - mask).round(eps), tt.matvec(mask, f).round(eps)
