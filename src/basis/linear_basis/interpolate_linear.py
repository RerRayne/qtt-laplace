# coding=UTF-8

from ttlaplace.zoperations import zaffine


def interpolate_linear(f, d):
    """
    # return tt-representation of a linear function c_0 + c_1 e_x + c_2 e_y = f(e_x, e_y)
    # f is called three times with args (0,0), (2**d-1, 0) and (0, 2**d-1)
    # Вот эта функция, собственно, вычисляет коэффициенты лин.зависимости якобиана от номера элемента.
    """
    n = 2**d
    c0 = f(0, 0)
    c1 = (f(n-1, 0) - c0) / (n-1.)
    c2 = (f(0, n-1) - c0) / (n-1.)
    return zaffine(c0, c1, c2, d)
