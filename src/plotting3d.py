"""
3D visualization: three orthogonal slices through a warm bubble.

Main entry point: make_bubble_3d(theta, grid, dark=False) -> base64 PNG.
"""
import base64
import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _as_3d(arr):
    if arr.ndim != 3:
        raise ValueError("expected 3D array, got shape " + str(arr.shape))
    return arr


def make_bubble_3d(theta, grid, dark=False,
                   theta_ref=None, title="3D warm bubble",
                   n_ticks=5):
    """
    Three orthogonal slices through the bubble centre.

    Parameters
    ----------
    theta : (nz, ny, nx) potential temperature
    grid : dict from make_grid3d
    dark : bool
    theta_ref : (nz,) or (nz, ny, nx) or None
        If provided, the anomaly theta - theta_ref is plotted.
    title : str
    n_ticks : int
        Number of ticks per axis.

    Returns
    -------
    str : base64-encoded PNG.
    """
    theta = _as_3d(np.asarray(theta, dtype=float))

    x_mc = grid["x_mc"]; y_mc = grid["y_mc"]; z_mc = grid["z_mc"]
    nz, ny, nx = theta.shape

    if theta.shape != (nz, ny, nx):
        raise ValueError("theta shape does not match grid")

    field = theta.copy()
    label = "theta, K"
    if theta_ref is not None:
        tr = np.asarray(theta_ref, dtype=float)
        if tr.shape == (nz,):
            tr = tr[:, None, None]
        if tr.shape != field.shape:
            raise ValueError("theta_ref shape mismatch")
        field = field - tr
        label = "theta - theta_ref, K"

    # centre indices
    i0 = nx // 2
    j0 = ny // 2
    k0 = nz // 2

    if dark:
        plt.style.use("dark_background")
        bg = "#0d1117"
        fg = "#e6edf3"
        grid_c = "#30363d"
    else:
        plt.style.use("default")
        bg = "#ffffff"
        fg = "#1f2328"
        grid_c = "#d0d7de"

    vmin = float(np.min(field))
    vmax = float(np.max(field))
    if vmax <= vmin:
        vmax = vmin + 1.0

    x_km = x_mc / 1000.0
    y_km = y_mc / 1000.0
    z_km = z_mc / 1000.0

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), facecolor=bg)
    fig.suptitle(title, color=fg, fontsize=12)

    # --- Panel 1: horizontal slice at z = z_mc[k0] (x-y plane) ---
    ax = axes[0]
    ax.set_facecolor(bg)
    slice_xy = field[k0, :, :]
    im = ax.pcolormesh(x_km, y_km, slice_xy,
                       cmap="YlOrRd", vmin=vmin, vmax=vmax,
                       shading="nearest")
    ax.set_xlabel("x, km", color=fg)
    ax.set_ylabel("y, km", color=fg)
    ax.set_title("z = %.1f km" % z_km[k0], color=fg, fontsize=10)
    ax.set_aspect("equal")
    ax.tick_params(colors=fg)
    for s in ax.spines.values():
        s.set_edgecolor(grid_c)

    # --- Panel 2: vertical slice at y = y_mc[j0] (x-z plane) ---
    ax = axes[1]
    ax.set_facecolor(bg)
    slice_xz = field[:, j0, :]     # (nz, nx)
    im = ax.pcolormesh(x_km, z_km, slice_xz,
                       cmap="YlOrRd", vmin=vmin, vmax=vmax,
                       shading="nearest")
    ax.set_xlabel("x, km", color=fg)
    ax.set_ylabel("z, km", color=fg)
    ax.set_title("y = %.1f km" % y_km[j0], color=fg, fontsize=10)
    ax.tick_params(colors=fg)
    for s in ax.spines.values():
        s.set_edgecolor(grid_c)

    # --- Panel 3: vertical slice at x = x_mc[i0] (y-z plane) ---
    ax = axes[2]
    ax.set_facecolor(bg)
    slice_yz = field[:, :, i0]     # (nz, ny)
    im = ax.pcolormesh(y_km, z_km, slice_yz,
                       cmap="YlOrRd", vmin=vmin, vmax=vmax,
                       shading="nearest")
    ax.set_xlabel("y, km", color=fg)
    ax.set_ylabel("z, km", color=fg)
    ax.set_title("x = %.1f km" % x_km[i0], color=fg, fontsize=10)
    ax.tick_params(colors=fg)
    for s in ax.spines.values():
        s.set_edgecolor(grid_c)

    fig.subplots_adjust(left=0.05, right=0.93, top=0.88, bottom=0.10,
                        wspace=0.30)

    cax = fig.add_axes([0.945, 0.15, 0.012, 0.70])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(label, color=fg)
    cb.ax.tick_params(colors=fg)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=bg)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_bubble_3d_bytes(theta, grid, **kwargs):
    """Same as make_bubble_3d but returns raw PNG bytes."""
    b64 = make_bubble_3d(theta, grid, **kwargs)
    return base64.b64decode(b64)
