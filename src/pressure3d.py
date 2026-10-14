"""
3D pressure projection on a collocation grid.

Divergence uses backward differences, gradient uses forward differences,
so that div(grad(phi)) is the NARROW Laplacian
    (phi[i+1] - 2 phi[i] + phi[i-1]) / dx^2 + ...
which is solved by Jacobi. This is consistent: div and grad are
adjoint (up to boundary terms), and the projection actually removes
the discrete divergence.
"""
import numpy as np


def solve_poisson_3d(rhs, dx, dy, dz, n_iter=12000, tol=1e-12):
    rhs = np.asarray(rhs, dtype=float)
    nz, ny, nx = rhs.shape
    ax = 1.0 / float(dx) ** 2
    ay = 1.0 / float(dy) ** 2
    az = 1.0 / float(dz) ** 2
    diag = 2.0 * (ax + ay + az)
    phi = np.zeros_like(rhs)
    for _ in range(n_iter):
        # Neumann: mirror
        xm = np.empty_like(phi); xm[:, :, 1:] = phi[:, :, :-1]; xm[:, :, 0] = phi[:, :, 0]
        xp = np.empty_like(phi); xp[:, :, :-1] = phi[:, :, 1:]; xp[:, :, -1] = phi[:, :, -1]
        ym = np.empty_like(phi); ym[:, 1:, :] = phi[:, :-1, :]; ym[:, 0, :] = phi[:, 0, :]
        yp = np.empty_like(phi); yp[:, :-1, :] = phi[:, 1:, :]; yp[:, -1, :] = phi[:, -1, :]
        zm = np.empty_like(phi); zm[1:, :, :] = phi[:-1, :, :]; zm[0, :, :] = phi[0, :, :]
        zp = np.empty_like(phi); zp[:-1, :, :] = phi[1:, :, :]; zp[-1, :, :] = phi[-1, :, :]
        phi_new = (ax * (xm + xp) + ay * (ym + yp) + az * (zm + zp) - rhs) / diag
        phi_new -= phi_new.mean()
        if float(np.max(np.abs(phi_new - phi))) < tol:
            phi = phi_new
            break
        phi = phi_new
    return phi


def _divergence(u, v, w, grid):
    """Backward differences everywhere (including boundaries)."""
    dx = grid["dx"]; dy = grid["dy"]; dz = grid["dz"]
    du = np.zeros_like(u)
    du[:, :, 1:] = (u[:, :, 1:] - u[:, :, :-1]) / dx[None, None, 1:]
    du[:, :, 0] = 0.0
    dv = np.zeros_like(v)
    dv[:, 1:, :] = (v[:, 1:, :] - v[:, :-1, :]) / dy[None, 1:, None]
    dv[:, 0, :] = 0.0
    dw = np.zeros_like(w)
    dw[1:, :, :] = (w[1:, :, :] - w[:-1, :, :]) / dz[1:, None, None]
    dw[0, :, :] = 0.0
    return du + dv + dw


def _gradient(phi, grid):
    """Forward differences everywhere (including boundaries)."""
    dx = grid["dx"]; dy = grid["dy"]; dz = grid["dz"]
    gx = np.zeros_like(phi)
    gx[:, :, :-1] = (phi[:, :, 1:] - phi[:, :, :-1]) / dx[None, None, :-1]
    gx[:, :, -1] = 0.0
    gy = np.zeros_like(phi)
    gy[:, :-1, :] = (phi[:, 1:, :] - phi[:, :-1, :]) / dy[None, :-1, None]
    gy[:, -1, :] = 0.0
    gz = np.zeros_like(phi)
    gz[:-1, :, :] = (phi[1:, :, :] - phi[:-1, :, :]) / dz[:-1, None, None]
    gz[-1, :, :] = 0.0
    return gx, gy, gz


def project_divergence_free(u, v, w, grid, dt, n_iter=12000, tol=1e-12):
    div_before = _divergence(u, v, w, grid)
    rhs = div_before / dt
    dx = grid["dx"]; dy = grid["dy"]; dz = grid["dz"]
    if not np.allclose(dx, dx[0]):
        raise ValueError("requires uniform dx")
    if not np.allclose(dy, dy[0]):
        raise ValueError("requires uniform dy")
    if not np.allclose(dz, dz[0]):
        raise ValueError("requires uniform dz")
    phi = solve_poisson_3d(rhs, dx[0], dy[0], dz[0], n_iter=n_iter, tol=tol)
    gx, gy, gz = _gradient(phi, grid)
    u_new = u - dt * gx
    v_new = v - dt * gy
    w_new = w - dt * gz
    div_after = _divergence(u_new, v_new, w_new, grid)
    return (u_new, v_new, w_new, phi,
            float(np.max(np.abs(div_before))),
            float(np.max(np.abs(div_after))))
