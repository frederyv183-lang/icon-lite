"""
3D model driver for the Flask /3d page.

All heavy lifting lives in the src modules: grid3d, state3d,
timestep3d, pressure3d, plotting3d. This file only wires them
together and produces base64 PNGs ready for the HTML template.

The grid is always UNIFORM in z (stretch=1.0) because the Jacobi
pressure solver requires a single dz.
"""
import base64

import numpy as np

from grid3d import make_grid3d
from state3d import make_initial_3d
from timestep3d import step_forward_3d
from pressure3d import project_divergence_free
from plotting3d import make_bubble_3d, make_bubble_3d_bytes


def _make_grid_and_state(nx, ny, nz,
                         x_total, y_total, z_top,
                         dtheta, qv_max, z0,
                         radius_x, radius_y, radius_z):
    g = make_grid3d(nx=nx, ny=ny, nz=nz,
                    x_total=x_total * 1000.0,
                    y_total=y_total * 1000.0,
                    z_top=z_top * 1000.0,
                    stretch=1.0)
    init = make_initial_3d(g,
                           dtheta=dtheta,
                           qv_max=qv_max / 1000.0,
                           z0_km=z0,
                           radius_x_km=radius_x,
                           radius_y_km=radius_y,
                           radius_z_km=radius_z)
    return g, init


def run_model_3d(nx, ny, nz,
                 x_total, y_total, z_top,
                 dtheta, qv_max, z0,
                 radius_x, radius_y, radius_z,
                 dark=False):
    """
    Static mode: render three orthogonal slices of the initial bubble.

    Returns a dict with base64 images and a few scalar diagnostics.
    """
    g, init = _make_grid_and_state(nx, ny, nz,
                                   x_total, y_total, z_top,
                                   dtheta, qv_max, z0,
                                   radius_x, radius_y, radius_z)
    theta = init["theta"]
    theta_ref = init["theta_ref_3d"]
    anomaly = theta - theta_ref

    img_initial = make_bubble_3d(theta, g, dark=dark,
                                 theta_ref=theta_ref,
                                 title="Initial warm bubble")

    return {
        "image_initial": img_initial,
        "max_dtheta": float(np.max(anomaly)),
        "min_dtheta": float(np.min(anomaly)),
        "max_qv_gkg": float(np.max(init["qv"])) * 1000.0,
        "n_cells": int(nx) * int(ny) * int(nz),
    }


def run_model_3d_rising(nx, ny, nz,
                        x_total, y_total, z_top,
                        dtheta, qv_max, z0,
                        radius_x, radius_y, radius_z,
                        dt, nsteps,
                        use_pressure=True,
                        n_snapshots=6,
                        dark=False):
    """
    Rising mode: integrate the model forward, then render a 2x3 grid
    of x-z slices at evenly spaced times.

    Returns a dict with base64 image plus diagnostics.
    """
    g, state = _make_grid_and_state(nx, ny, nz,
                                    x_total, y_total, z_top,
                                    dtheta, qv_max, z0,
                                    radius_x, radius_y, radius_z)

    theta_ref = state["theta_ref_3d"]
    j0 = g["ny"] // 2

    # first snapshot
    snapshots = [(0.0, state["theta"].copy())]
    every = max(1, int(nsteps) // int(n_snapshots))

    t = 0.0
    for step in range(1, int(nsteps) + 1):
        state = step_forward_3d(state, g, dt=dt)
        if use_pressure:
            u, v, w, phi, db, da = project_divergence_free(
                state["u"], state["v"], state["w"], g, dt=dt,
            )
            state["u"] = u
            state["v"] = v
            state["w"] = w
        t += dt
        if step % every == 0 and len(snapshots) < int(n_snapshots):
            snapshots.append((t, state["theta"].copy()))

    img_rising = _draw_rising_grid(snapshots, g, theta_ref, j0,
                                   dark=dark)

    max_w = float(np.max(np.abs(state["w"])))
    return {
        "image_rising": img_rising,
        "max_w": max_w,
        "final_t": t,
        "n_snapshots": len(snapshots),
        "use_pressure": bool(use_pressure),
    }


def _draw_rising_grid(snapshots, g, theta_ref, j0, dark=False):
    """2x3 grid of x-z slices of theta - theta_ref."""
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if dark:
        plt.style.use("dark_background")
        bg = "#0d1117"; fg = "#e6edf3"; grid_c = "#30363d"
    else:
        plt.style.use("default")
        bg = "#ffffff"; fg = "#1f2328"; grid_c = "#d0d7de"

    x_km = g["x_mc"] / 1000.0
    z_km = g["z_mc"] / 1000.0

    vmax = max(
        float(np.max(np.abs(th - theta_ref)))
        for _, th in snapshots
    ) or 1.0

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.0), facecolor=bg)
    fig.suptitle("3D warm bubble rising (x-z slice, y = centre)",
                 color=fg, fontsize=12)

    for ax, (t_snap, theta_snap) in zip(axes.ravel(), snapshots[:6]):
        ax.set_facecolor(bg)
        anom = theta_snap - theta_ref
        slice_xz = anom[:, j0, :]
        im = ax.pcolormesh(x_km, z_km, slice_xz,
                           cmap="YlOrRd", vmin=0.0, vmax=vmax,
                           shading="nearest")
        ax.set_xlabel("x, km", color=fg)
        ax.set_ylabel("z, km", color=fg)
        ax.set_title("t = %.0f s" % t_snap, color=fg, fontsize=10)
        ax.tick_params(colors=fg)
        for s in ax.spines.values():
            s.set_edgecolor(grid_c)

    fig.subplots_adjust(left=0.05, right=0.92, top=0.90, bottom=0.08,
                        wspace=0.25, hspace=0.35)
    cax = fig.add_axes([0.935, 0.15, 0.012, 0.70])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("theta - theta_ref, K", color=fg)
    cb.ax.tick_params(colors=fg)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=bg)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")
