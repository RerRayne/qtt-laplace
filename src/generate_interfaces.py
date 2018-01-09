# coding=utf-8

__author__ = 'l.markeeva'
import numpy as np
import tt


def getSew(d, side, inversed=False):
    """
    This method returns an interface matrix for sewing between two subdomains
    :param d:
    :param side:
    :param inversed:
    :return:
    """
    side = side.upper()
    B = R = L = T = 0
    LLC = LRC = ULC = URC = 0
    if side == "TOP":
        T = 1
    if side == "BOTTOM":
        B = 1
    if side == "LEFT":
        L = 1
    if side == "RIGHT":
        R = 1
    if side == "LLC":
        LLC = 1
    if side == "LRC":
        LRC = 1
    if side == "ULC":
        ULC = 1
    if side == "URC":
        URC = 1

    if B + T + L + R == 1:
        # side case
        if inversed:
            core = np.array([[L, B, T, R], [B, R, L, T]]).reshape([1, 2, 4, 1], order='F')
        else:
            core = np.array([[B, R, L, T], [L, B, T, R]]).reshape([1, 2, 4, 1], order='F')
        return tt.matrix.from_list([core] * d)

    if LLC + LRC + ULC + URC == 1:
        core = np.array([[LLC, LRC, ULC, URC]]).reshape([1, 1, 4, 1], order='F')
        return tt.matrix.from_list([core] * d)

    raise ValueError("Valid inputs are TOP, BOTTOM, LEFT, RIGHT, LLC, LRC, ULC, URC")


def getP(d, side_m, side_p):
    """
    Get sewing matrices
    :param d:
    :param side_m:
    :param side_p:
    :return:
    """
    UpsilonM = getSew(d, side=side_m, inversed=False)
    UpsilonP = getSew(d, side=side_p, inversed=True)

    Pmp = UpsilonM.T * UpsilonP
    Ppm = UpsilonP.T * UpsilonM
    Pmm = -(UpsilonM.T * UpsilonM)
    Ppp = -(UpsilonP.T * UpsilonP)

    return Pmp, Ppm, Pmm, Ppp