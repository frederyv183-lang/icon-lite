"""
2D diagnostics of storms: CAPE at every horizontal point.

For each column x we call compute_cape from storm.py and
collect fields CAPE(x, z), LFC(x), EL(x).
"""
import numpy as np

from storm import compute_cape, compute_li, storm_index


def compute_cape_2d(z_mc, theta_2d, qv_2d, exner_2d,
                    k_surf_max=500.0):
    """
    CAPE for each column in a 2D field.

    Returns dict:
        cape_2d   : (nz, nx)   CAPE contribution per layer and column
        cape_col  : (nx,)      total CAPE in each column
        lfc_x, el_x, lcl_x, z_trop_x : (nx,)  level arrays (nan if absent)
        max_cape, mean_cape : float
        argmax_x : int         index of the column with max CAPE
    """
    z_mc = np.asarray(z_mc, dtype=float)
    theta_2d = np.asarray(theta_2d, dtype=float)
    qv_2d = np.asarray(qv_2d, dtype=float)
    exner_2d = np.asarray(exner_2d, dtype=float)
    nz, nx = theta_2d.shape

    k_surf = int(np.searchsorted(z_mc, k_surf_max, side="right"))
    if k_surf < 1:
        k_surf = 1

    cape_2d = np.zeros((nz, nx))
    cape_col = np.zeros(nx)
    lfc_x = np.full(nx, np.nan)
    el_x = np.full(nx, np.nan)
    lcl_x = np.full(nx, np.nan)
    z_trop_x = np.full(nx, np.nan)

    for j in range(nx):
        theta_col = theta_2d[:, j]
        qv_col = qv_2d[:, j]
        exner_col = exner_2d[:, j]

        theta_part = float(np.max(theta_col[:k_surf]))
        qv_part = float(np.max(qv_col[:k_surf]))

        try:
            res = compute_cape(z_mc, theta_col, qv_col, exner_col,
                               theta_part_surf=theta_part,
                               qv_part_surf=qv_part)
        except Exception:
            continue

        cape_2d[:, j] = res["cape_profile"]
        cape_col[j] = res["cape_Jkg"]
        if res["lfc_m"] is not None:
            lfc_x[j] = res["lfc_m"]
        if res["el_m"] is not None:
            el_x[j] = res["el_m"]
        lcl_x[j] = res["z_lcl_m"]
        z_trop_x[j] = res.get("z_trop_m", np.nan)

    max_cape = float(np.max(cape_col))
    mean_cape = float(np.mean(cape_col))
    argmax_x = int(np.argmax(cape_col))

    return {
        "cape_2d": cape_2d,
        "cape_col": cape_col,
        "lfc_x": lfc_x,
        "el_x": el_x,
        "lcl_x": lcl_x,
        "z_trop_x": z_trop_x,
        "max_cape": max_cape,
        "mean_cape": mean_cape,
        "argmax_x": argmax_x,
    }


def storm2d_metrics(cape_2d_result, max_w):
    """
    Summary metrics for /storm2d.
    """
    cape_col = cape_2d_result["cape_col"]
    cape_max = float(np.max(cape_col))
    cape_mean = float(np.mean(cape_col))
    cape_min = float(np.min(cape_col))

    lfc_valid = cape_2d_result["lfc_x"][~np.isnan(cape_2d_result["lfc_x"])]
    el_valid = cape_2d_result["el_x"][~np.isnan(cape_2d_result["el_x"])]

    lfc_min = float(np.min(lfc_valid)) if lfc_valid.size else None
    el_max = float(np.max(el_valid)) if el_valid.size else None

    traffic = storm_index(cape_max, 0.0, max_w)

    return {
        "cape_max": cape_max,
        "cape_mean": cape_mean,
        "cape_min": cape_min,
        "lfc_min": lfc_min,
        "el_max": el_max,
        "traffic": traffic,
    }
