"""
3D visualization: three orthogonal slices through a field.

Main entry points:
  - make_field_3d(field, grid, ...)  - universal
  - make_bubble_3d(theta, grid, ...) - backward-compat wrapper
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


def make_field_3d(field, grid, dark=False,
                  field_ref=None, title="3D field",
                  cmap="YlOrRd", units="",
                  symmetric=False, diverging_cmap="RdBu_r"):
    """
    Three orthogonal slices through the centre of a 3D field.

    symmetric=True -> diverging colormap centred at 0 (good for w).
    Returns base64 PNG.
    """
    field = _as_3d(np.asarray(field, dtype=float))
    x_mc = grid["x_mc"]; y_mc = grid["y_mc"]; z_mc = grid["z_mc"]
    nz, ny, nx = field.shape

    label = title if not units else (title + ", " + units)

    if field_ref is not None:
        tr = np.asarray(field_ref, dtype=float)
        if tr.shape == (nz,):
            tr = tr[:, None, None]
        if tr.shape != field.shape:
            raise ValueError("field_ref shape mismatch")
        field = field - tr
        label = (title + " - ref") if not units else (title + " - ref, " + units)

    i0 = nx // 2; j0 = ny // 2; k0 = nz // 2

    if dark:
        plt.style.use("dark_background")
        bg = "#0d1117"; fg = "#e6edf3"; grid_c = "#30363d"
    else:
        plt.style.use("default")
        bg = "#ffffff"; fg = "#1f2328"; grid_c = "#d0d7de"

    if symmetric:
        vmax = float(np.max(np.abs(field))) or 1.0
        vmin = -vmax
        use_cmap = diverging_cmap
    else:
        vmin = float(np.min(field))
        vmax = float(np.max(field))
        if vmax <= vmin:
            vmax = vmin + 1.0
        use_cmap = cmap

    x_km = x_mc / 1000.0
    y_km = y_mc / 1000.0
    z_km = z_mc / 1000.0

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), facecolor=bg)
    fig.suptitle(title, color=fg, fontsize=12)

    ax = axes[0]
    ax.set_facecolor(bg)
    im = ax.pcolormesh(x_km, y_km, field[k0, :, :],
                       cmap=use_cmap, vmin=vmin, vmax=vmax,
                       shading="nearest")
    ax.set_xlabel("x, km", color=fg)
    ax.set_ylabel("y, km", color=fg)
    ax.set_title("z = %.1f km" % z_km[k0], color=fg, fontsize=10)
    ax.set_aspect("equal")
    ax.tick_params(colors=fg)
    for s in ax.spines.values():
        s.set_edgecolor(grid_c)

    ax = axes[1]
    ax.set_facecolor(bg)
    im = ax.pcolormesh(x_km, z_km, field[:, j0, :],
                       cmap=use_cmap, vmin=vmin, vmax=vmax,
                       shading="nearest")
    ax.set_xlabel("x, km", color=fg)
    ax.set_ylabel("z, km", color=fg)
    ax.set_title("y = %.1f km" % y_km[j0], color=fg, fontsize=10)
    ax.tick_params(colors=fg)
    for s in ax.spines.values():
        s.set_edgecolor(grid_c)

    ax = axes[2]
    ax.set_facecolor(bg)
    im = ax.pcolormesh(y_km, z_km, field[:, :, i0],
                       cmap=use_cmap, vmin=vmin, vmax=vmax,
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


def make_field_3d_bytes(field, grid, **kwargs):
    return base64.b64decode(make_field_3d(field, grid, **kwargs))


def make_bubble_3d(theta, grid, dark=False,
                   theta_ref=None, title="3D warm bubble"):
    return make_field_3d(theta, grid, dark=dark,
                         field_ref=theta_ref,
                         title=title,
                         cmap="YlOrRd", units="K",
                         symmetric=False)


def make_bubble_3d_bytes(theta, grid, **kwargs):
    return base64.b64decode(make_bubble_3d(theta, grid, **kwargs))
