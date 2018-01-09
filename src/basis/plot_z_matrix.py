import matplotlib.pyplot as plt

import numpy as np

from scipy.spatial import Delaunay

import tt

from ttlaplace.zoperations import zkronv


def plot_patch(mesh_container, x, d, vmin=0, vmax=1, contours=True):
    px = zkronv(tt.xfun(2, d), tt.ones(2, d))
    py = zkronv(tt.ones(2, d), tt.xfun(2, d))
    px = px.full().flatten('F').astype(np.int)
    py = py.full().flatten('F').astype(np.int)

    pts = np.array([mesh_container._pts(ex, ey) for ex, ey in zip(px, py)])
    tri = Delaunay(pts)
    plt.tripcolor(pts[:,0], pts[:,1], tri.simplices.copy(), x, shading='gouraud', vmin=vmin, vmax=vmax)
    if contours:
        plt.tricontour(pts[:,0], pts[:,1], tri.simplices.copy(), x, np.linspace(vmin, vmax, 11), colors='k')
    d = mesh_container.d
    p1 = mesh_container._pts(0, 0)
    p2 = mesh_container._pts(2 ** d - 1, 0)
    p3 = mesh_container._pts(2 ** d - 1, 2 ** d - 1)
    p4 = mesh_container._pts(0, 2 ** d - 1)
    plt.plot([p1[0], p2[0], p3[0], p4[0]],
             [p1[1], p2[1], p3[1], p4[1]], 'm')

