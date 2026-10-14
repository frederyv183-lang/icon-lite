"""
3D initial conditions: spherical warm bubble. All fields (nz, ny, nx).
"""
import numpy as np


def _reference_atmosphere(z_mc, theta_surf=300.0, dtheta_dz=0.0035):
    return float(theta_surf) + float(dtheta_dz) * np.asarray(z_mc)


def make_initial_3d(grid, theta_ref=None,
                    dtheta=2.0, qv_max=0.0,
                    x0_km=None, y0_km=None, z0_km=3.0,
                    radius_x_km=5.0, radius_y_km=5.0, radius_z_km=1.5,
                    theta_surf=300.0):
    x_mc = grid["x_mc"]; y_mc = grid["y_mc"]; z_mc = grid["z_mc"]
    nx = grid["nx"]; ny = grid["ny"]; nz = grid["nz"]

    x_total = float(x_mc[-1] + (x_mc[1] - x_mc[0]) * 0.5)
    y_total = float(y_mc[-1] + (y_mc[1] - y_mc[0]) * 0.5)

    x0 = float(x0_km) * 1000.0 if x0_km is not None else x_total * 0.5
    y0 = float(y0_km) * 1000.0 if y0_km is not None else y_total * 0.5
    z0 = float(z0_km) * 1000.0

    rx = float(radius_x_km) * 1000.0
    ry = float(radius_y_km) * 1000.0
    rz = float(radius_z_km) * 1000.0

    X = x_mc[None, None, :]
    Y = y_mc[None, :, None]
    Z = z_mc[:, None, None]
    r2 = ((X - x0) / rx) ** 2 + ((Y - y0) / ry) ** 2 + ((Z - z0) / rz) ** 2
    shape = np.exp(-r2)

    if theta_ref is None:
        theta_ref_1d = _reference_atmosphere(z_mc, theta_surf=theta_surf)
    else:
        theta_ref_1d = np.asarray(theta_ref)
        if theta_ref_1d.shape != (nz,):
            raise ValueError("theta_ref must have shape (nz,)")

    theta_ref_3d = np.broadcast_to(
        theta_ref_1d[:, None, None], (nz, ny, nx)).copy()

    theta = theta_ref_3d + float(dtheta) * shape
    qv = float(qv_max) * shape

    H = 8000.0; rho_surf = 1.2
    rho_1d = rho_surf * np.exp(-z_mc / H)
    rho = np.broadcast_to(rho_1d[:, None, None], (nz, ny, nx)).copy()

    qc = np.zeros((nz, ny, nx), dtype=float)
    u = np.zeros((nz, ny, nx), dtype=float)
    v = np.zeros((nz, ny, nx), dtype=float)
    w = np.zeros((nz, ny, nx), dtype=float)

    return {
        "theta": theta, "qv": qv, "qc": qc, "rho": rho,
        "u": u, "v": v, "w": w, "theta_ref_3d": theta_ref_3d,
    }
