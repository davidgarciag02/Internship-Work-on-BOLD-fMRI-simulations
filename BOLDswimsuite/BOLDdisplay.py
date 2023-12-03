from . import BOLDvessel
from mayavi import mlab
import matplotlib.pyplot as plt
from typing import List
import numpy as np

def mlab_plot_infinite_cylinder_3d(vessel: BOLDvessel.InfiniteCylinder3DNumba, size: float, extend: bool=True):

    radius_size = (np.sqrt(3)*size)/2 if extend else size/2
    axial_points_per_mm = 100/size

    curr_pos = np.array(vessel.origin)
    while np.all(np.logical_and(curr_pos < radius_size, curr_pos > -radius_size)):
        curr_pos += vessel.normal_vector/axial_points_per_mm
    axial_point_pos = np.array(curr_pos)

    curr_pos = vessel.origin
    while np.all(np.logical_and(curr_pos < radius_size, curr_pos > -radius_size)):
        curr_pos -= vessel.normal_vector/axial_points_per_mm
    axial_point_neg = np.array(curr_pos)

    axial_points = np.vstack([axial_point_neg, axial_point_pos])

    X = axial_points[:, 0]
    Y = axial_points[:, 1]
    Z = axial_points[:, 2]

    mlab.plot3d(X, Y, Z, tube_radius=vessel.diameter/2, color=(1.,0.,0.))

def mlab_plot_infinite_cylinder_3d_list(vessels: List[BOLDvessel.InfiniteCylinder3DNumba], size: float, extend: bool=True):
    for vessel in vessels:
        mlab_plot_infinite_cylinder_3d(
            vessel=vessel, 
            size=size, 
            extend=extend
        )

def mlab_plot_sphere_3d_list(vessels: List[BOLDvessel.Sphere3DNumba]):
    
    X = np.zeros(len(vessels), dtype=float)
    Y = np.zeros(len(vessels), dtype=float)
    Z = np.zeros(len(vessels), dtype=float)
    D = np.zeros(len(vessels), dtype=float)
    
    for i, vessel in enumerate(vessels):
        X[i] = vessel.origin[0] 
        Y[i] = vessel.origin[1]
        Z[i] = vessel.origin[2]
        D[i] = vessel.diameter

    mlab.points3d(X, Y, Z, D, scale_factor=1, color=(1.,0.,0.))

def mlab_plot_sphere_3d(vessel: BOLDvessel.Sphere3DNumba):
    mlab_plot_sphere_3d_list(vessels=[vessel])


def mlab_plot_dBz_grid_3d(dBz_grid: np.ndarray):

    N = dBz_grid.shape[0]

    mlab.contour3d(dBz_grid, contours=8, transparent=True)
    mlab.outline(color=(0, 0, 0), line_width=5, extent=[0, N, 0, N, 0, N])

def mlab_plot_is_IV_grid_3d(is_IV_grid: np.ndarray):

    is_IV_grid = is_IV_grid.astype(float)

    N = is_IV_grid.shape[0]

    mlab.contour3d(is_IV_grid, contours=2, transparent=False)
    mlab.outline(color=(0, 0, 0), line_width=5, extent=[0, N, 0, N, 0, N])

def matplotlib_plot_infinite_cylinder_2d_list(vessels: List[BOLDvessel.InfiniteCylinder2DNumba]):
    
    fig = plt.gcf()
    ax = fig.gca()
    for vessel in vessels:
        circle = plt.Circle(vessel.origin, vessel.diameter/2)
        ax.add_patch(circle)

