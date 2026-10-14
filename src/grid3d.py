"""
3D grid: regular in (x, y), stretched in z (same stretch as 1D/2D).
Arrays are ordered (nz, ny, nx) for fields, (nz+1, ny, nx) for w.
"""
import numpy as np


def _stretch_z(nz, z_top, stretch):
    """Stretched vertical levels: geometric growth of dz towards top."""
    if nz < 2:
        return np.array([0.0, float(z_top)]), np.array([float(z_top)])
    # Ratio such that sum of geometric series of dz = z_top
    # dz_k = dz0 * stretch^k, k = 0..nz-1
    # dz0 * (stretch^nz - 1) / (stretch - 1) = z_top
    dz0 = z_top * (stretch - 1.0) / (stretch ** nz - 1.0)
    k = np.arange(nz)
    dz = dz0 * stretch ** k
    z_ifc = np.concatenate([[0.0], np.cumsum(dz)])
    z_ifc[-1] = z_top  # exact
    return z_ifc, dz


def make_grid3d(nx, ny, nz, x_total=40000.0, y_total=40000.0,
                z_top=20000.0, stretch=1.15):
    """
    Build a 3D Cartesian grid.

    Parameters
    ----------
    nx, ny, nz : int
    x_total, y_total : float, m
    z_top : float, m
    stretch : float, dz ratio between adjacent levels

    Returns
    -------
    dict with keys:
      x_mc, y_mc, z_mc       : (nx,), (ny,), (nz,)        cell centers
      x_edge, y_edge, z_ifc  : (nx+1,), (ny+1,), (nz+1,)  cell edges
      dx, dy, dz             : (nx,), (ny,), (nz,)        cell sizes
      nx, ny, nz             : int
      wgtfac_c               : (nz,)  weight for linear interp to cell centers
    """
    if nx < 2 or ny < 2 or nz < 2:
        raise ValueError("nx, ny, nz must be >= 2")

    dx_val = float(x_total) / float(nx)
    dy_val = float(y_total) / float(ny)

    x_edge = np.linspace(0.0, float(x_total), nx + 1)
    y_edge = np.linspace(0.0, float(y_total), ny + 1)

    x_mc = 0.5 * (x_edge[:-1] + x_edge[1:])
    y_mc = 0.5 * (y_edge[:-1] + y_edge[1:])

    z_ifc, dz = _stretch_z(nz, float(z_top), float(stretch))
    z_mc = 0.5 * (z_ifc[:-1] + z_ifc[1:])

    dx = np.full(nx, dx_val)
    dy = np.full(ny, dy_val)

    # wgtfac_c: weight of the lower interface in center interpolation
    # z_mc[k] = wgtfac_c[k] * z_ifc[k] + (1 - wgtfac_c[k]) * z_ifc[k+1]
    wgtfac_c = (z_ifc[1:] - z_mc) / (z_ifc[1:] - z_ifc[:-1])
    wgtfac_c = np.clip(wgtfac_c, 0.0, 1.0)

    return {
        "x_mc": x_mc, "y_mc": y_mc, "z_mc": z_mc,
        "x_edge": x_edge, "y_edge": y_edge, "z_ifc": z_ifc,
        "dx": dx, "dy": dy, "dz": dz,
        "nx": int(nx), "ny": int(ny), "nz": int(nz),
        "wgtfac_c": wgtfac_c,
    }
