"""
One forward step of the 3D model (NO pressure solver).

Physics:
  - Upwind advection of theta, qv, u, v, w
  - Buoyancy: dw/dt = g * (theta - theta_ref) / theta_ref  (on w interfaces)

Field layout: all prognostic fields (nz, ny, nx), w on (nz+1, ny, nx).

This is an incomplete model: the flow is not divergence-free,
so the bubble will spread over time. The purpose is to demonstrate
the buoyant rise and to give a stepping stone for adding the
pressure projection later.
"""
import warnings

import numpy as np

G = 9.81


def _advect_upwind(field, u, v, w_c, grid, dt):
    """
    Upwind advection of a scalar field (nz, ny, nx).

    u, v at cell centers (nz, ny, nx)
    w_c at cell centers (nz, ny, nx), interpolated from w on interfaces
    """
    nx = grid["nx"]; ny = grid["ny"]; nz = grid["nz"]
    dx = grid["dx"][None, None, :]
    dy = grid["dy"][None, :, None]
    dz = grid["dz"][:, None, None]

    out = field.copy()

    # x-direction
    fx = np.zeros_like(field)
    u_pos = np.maximum(u, 0.0)
    u_neg = np.minimum(u, 0.0)
    fx[:, :, 1:] += u_pos[:, :, 1:] * (field[:, :, 1:] - field[:, :, :-1]) / dx[:, :, 1:]
    fx[:, :, :-1] += u_neg[:, :, :-1] * (field[:, :, 1:] - field[:, :, :-1]) / dx[:, :, :-1]

    # y-direction
    fy = np.zeros_like(field)
    v_pos = np.maximum(v, 0.0)
    v_neg = np.minimum(v, 0.0)
    fy[:, 1:, :] += v_pos[:, 1:, :] * (field[:, 1:, :] - field[:, :-1, :]) / dy[:, 1:, :]
    fy[:, :-1, :] += v_neg[:, :-1, :] * (field[:, 1:, :] - field[:, :-1, :]) / dy[:, :-1, :]

    # z-direction
    fz = np.zeros_like(field)
    w_pos = np.maximum(w_c, 0.0)
    w_neg = np.minimum(w_c, 0.0)
    fz[1:, :, :] += w_pos[1:, :, :] * (field[1:, :, :] - field[:-1, :, :]) / dz[1:, :, :]
    fz[:-1, :, :] += w_neg[:-1, :, :] * (field[1:, :, :] - field[:-1, :, :]) / dz[:-1, :, :]

    out = field - dt * (fx + fy + fz)
    return out


def step_forward_3d(state, grid, dt,
                    use_advection=True,
                    use_buoyancy=True):
    """
    Advance the 3D state by one time step.

    Parameters
    ----------
    state : dict with keys
        theta (nz, ny, nx), qv (nz, ny, nx), u (nz, ny, nx),
        v (nz, ny, nx), w (nz+1, ny, nx), rho (nz, ny, nx),
        theta_ref_3d (nz, ny, nx)
    grid : dict from make_grid3d
    dt : float
    use_advection, use_buoyancy : bool
        For testing individual terms.

    Returns
    -------
    dict with updated theta, qv, u, v, w  (rho, theta_ref_3d unchanged)
    """
    theta = state["theta"]
    qv = state["qv"]
    u = state["u"]
    v = state["v"]
    w = state["w"]
    theta_ref = state["theta_ref_3d"]

    nz = grid["nz"]; ny = grid["ny"]; nx = grid["nx"]

    # CFL check
    max_u = float(np.max(np.abs(u))) if u.size else 0.0
    max_v = float(np.max(np.abs(v))) if v.size else 0.0
    max_w = float(np.max(np.abs(w))) if w.size else 0.0
    dx_min = float(np.min(grid["dx"]))
    dy_min = float(np.min(grid["dy"]))
    dz_min = float(np.min(grid["dz"]))
    cfl = dt * max(max_u / dx_min, max_v / dy_min, max_w / dz_min)
    if cfl > 0.9:
        warnings.warn(
            "CFL=%.3f > 0.9; reduce dt or coarsen the grid" % cfl,
            RuntimeWarning, stacklevel=2,
        )

    # w at cell centers (nz, ny, nx)
    wgt = grid["wgtfac_c"][:, None, None]
    w_c = wgt * w[:-1, :, :] + (1.0 - wgt) * w[1:, :, :]

    if use_advection:
        theta_new = _advect_upwind(theta, u, v, w_c, grid, dt)
        qv_new = _advect_upwind(qv, u, v, w_c, grid, dt)
        u_new = _advect_upwind(u, u, v, w_c, grid, dt)
        v_new = _advect_upwind(v, u, v, w_c, grid, dt)
        w_c_new = _advect_upwind(w_c, u, v, w_c, grid, dt)
    else:
        theta_new = theta.copy()
        qv_new = qv.copy()
        u_new = u.copy()
        v_new = v.copy()
        w_c_new = w_c.copy()

    # Buoyancy on w (interfaces). We use the cell-center anomaly and
    # linearly interpolate it to interfaces using wgtfac_c.
    if use_buoyancy:
        b = G * (theta_new - theta_ref) / theta_ref   # (nz, ny, nx)
        # interface values: average adjacent cells
        b_ifc = np.zeros((nz + 1, ny, nx), dtype=float)
        b_ifc[1:-1, :, :] = 0.5 * (b[:-1, :, :] + b[1:, :, :])
        b_ifc[0, :, :] = b[0, :, :]
        b_ifc[-1, :, :] = b[-1, :, :]
        w_new = w + dt * b_ifc
    else:
        # rebuild w from cell centers if we only advected w_c
        w_new = np.zeros((nz + 1, ny, nx), dtype=float)
        w_new[1:-1] = 0.5 * (w_c_new[:-1] + w_c_new[1:])
        w_new[0] = w_c_new[0]
        w_new[-1] = w_c_new[-1]

    if use_advection and use_buoyancy:
        # after advection we keep w_c_new in mind but buoyancy is applied
        # to the original w; below we rebuild the interface w from the
        # advected center values plus buoyancy increment.
        # Simple approach: project w_c_new back to interfaces and add buoyancy.
        w_advected = np.zeros((nz + 1, ny, nx), dtype=float)
        w_advected[1:-1] = 0.5 * (w_c_new[:-1] + w_c_new[1:])
        w_advected[0] = w_c_new[0]
        w_advected[-1] = w_c_new[-1]
        w_new = w_advected + dt * b_ifc

    return {
        "theta": theta_new,
        "qv": qv_new,
        "u": u_new,
        "v": v_new,
        "w": w_new,
        "rho": state["rho"],
        "theta_ref_3d": theta_ref,
    }
