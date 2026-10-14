"""
One forward step (advection + buoyancy). All fields on centres.
"""
import warnings
import numpy as np

G = 9.81


def _advect_upwind(field, u, v, w, grid, dt):
    dx = grid["dx"][None, None, :]
    dy = grid["dy"][None, :, None]
    dz = grid["dz"][:, None, None]

    fx = np.zeros_like(field)
    up = np.maximum(u, 0.0); un = np.minimum(u, 0.0)
    fx[:, :, 1:] += up[:, :, 1:] * (field[:, :, 1:] - field[:, :, :-1]) / dx[:, :, 1:]
    fx[:, :, :-1] += un[:, :, :-1] * (field[:, :, 1:] - field[:, :, :-1]) / dx[:, :, :-1]

    fy = np.zeros_like(field)
    vp = np.maximum(v, 0.0); vn = np.minimum(v, 0.0)
    fy[:, 1:, :] += vp[:, 1:, :] * (field[:, 1:, :] - field[:, :-1, :]) / dy[:, 1:, :]
    fy[:, :-1, :] += vn[:, :-1, :] * (field[:, 1:, :] - field[:, :-1, :]) / dy[:, :-1, :]

    fz = np.zeros_like(field)
    wp = np.maximum(w, 0.0); wn = np.minimum(w, 0.0)
    fz[1:, :, :] += wp[1:, :, :] * (field[1:, :, :] - field[:-1, :, :]) / dz[1:, :, :]
    fz[:-1, :, :] += wn[:-1, :, :] * (field[1:, :, :] - field[:-1, :, :]) / dz[:-1, :, :]

    return field - dt * (fx + fy + fz)


def step_forward_3d(state, grid, dt,
                    use_advection=True, use_buoyancy=True,
                    use_pressure=False, n_iter_pressure=3000):
    theta = state["theta"]; qv = state["qv"]
    u = state["u"]; v = state["v"]; w = state["w"]
    theta_ref = state["theta_ref_3d"]

    max_u = float(np.max(np.abs(u)))
    max_v = float(np.max(np.abs(v)))
    max_w = float(np.max(np.abs(w)))
    cfl = dt * max(max_u / float(np.min(grid["dx"])),
                   max_v / float(np.min(grid["dy"])),
                   max_w / float(np.min(grid["dz"])))
    if cfl > 0.9:
        warnings.warn("CFL=%.3f > 0.9" % cfl, RuntimeWarning, stacklevel=2)

    if use_advection:
        theta_new = _advect_upwind(theta, u, v, w, grid, dt)
        qv_new    = _advect_upwind(qv,    u, v, w, grid, dt)
        u_new     = _advect_upwind(u,     u, v, w, grid, dt)
        v_new     = _advect_upwind(v,     u, v, w, grid, dt)
        w_new     = _advect_upwind(w,     u, v, w, grid, dt)
    else:
        theta_new = theta.copy(); qv_new = qv.copy()
        u_new = u.copy(); v_new = v.copy(); w_new = w.copy()

    if use_buoyancy:
        b = G * (theta_new - theta_ref) / theta_ref
        w_new = w_new + dt * b

    if use_pressure:
        try:
            from pressure3d import project_divergence_free
            u_new, v_new, w_new, _phi, _db, _da = project_divergence_free(
                u_new, v_new, w_new, grid, dt,
                n_iter=n_iter_pressure,
            )
        except ImportError:
            pass

    return {
        "theta": theta_new, "qv": qv_new,
        "u": u_new, "v": v_new, "w": w_new,
        "rho": state["rho"], "theta_ref_3d": theta_ref,
    }
