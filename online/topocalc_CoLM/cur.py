import numpy as np


def cur(dem, dx, dy):
    """ArcGIS method to calcuate curvature"""
    # Pad the dem
    dem_pad = np.pad(dem, pad_width=1, mode='edge')

    # Second-order finite difference in the y direction
    E = ((dem_pad[2:, 1:-1] +dem_pad[:-2, 1:-1])/2- dem_pad[1:-1,1:-1])/ (dy**2)

    # Second-order finite difference in the x direction
    D = ((dem_pad[1:-1, 2:] + dem_pad[1:-1, :-2] )/2- dem_pad[1:-1,1:-1])/ (dx**2)
    curvature = -2*(D+E)*100

    return curvature
