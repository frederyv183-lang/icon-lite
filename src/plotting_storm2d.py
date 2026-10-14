"""
2D visualization of storm diagnostics: CAPE(x, z) map + w(x) profile.
"""
import io
import base64

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
#  3D-визуализация: CAPE(x, z) как поверхность
# ---------------------------------------------------------------------

def make_storm2d_map_3d(x_mc, z_mc, cape_2d, dark=False,
                        cape_max=None, cape_mean=None,
                        lfc_x=None, el_x=None, traffic=None,
                        elev=28.0, azim=-60.0):
    """
    3D-поверхность CAPE(x, z). Возвращает bytes PNG.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if dark:
        plt.style.use("dark_background")
        bg = "#0d1117"
        fg = "#e6edf3"
        grid = "#30363d"
    else:
        plt.style.use("default")
        bg = "#ffffff"
        fg = "#1f2328"
        grid = "#d0d7de"

    x_km = x_mc / 1000.0
    z_km = z_mc / 1000.0

    cmax = float(np.max(cape_2d)) if cape_2d.size > 0 else 1.0
    if cmax <= 0.0:
        cmax = 1.0

    fig = plt.figure(figsize=(10.0, 6.4), facecolor=bg)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(bg)

    X, Z = np.meshgrid(x_km, z_km)  # (nz, nx)

    surf = ax.plot_surface(
        X, Z, cape_2d,
        cmap="YlOrRd", vmin=0.0, vmax=cmax,
        linewidth=0, antialiased=True, alpha=0.95,
        rstride=1, cstride=1,
    )

    if lfc_x is not None and el_x is not None:
        lfc_km = np.where(np.isfinite(lfc_x), lfc_x / 1000.0, np.nan)
        el_km = np.where(np.isfinite(el_x), el_x / 1000.0, np.nan)
        valid = np.isfinite(lfc_km) & np.isfinite(el_km)
        if np.any(valid):
            n_valid = int(np.sum(valid))
            ax.scatter(lfc_km[valid], np.zeros(n_valid), np.zeros(n_valid),
                       c="#2563eb", s=8, depthshade=False, label="LFC")
            ax.scatter(el_km[valid], np.zeros(n_valid), np.zeros(n_valid),
                       c="#7c3aed", s=8, depthshade=False, label="EL")
            ax.legend(loc="upper right", fontsize=8,
                      facecolor=bg, edgecolor=grid, labelcolor=fg)

    ax.set_xlabel("x, км", color=fg, labelpad=8)
    ax.set_ylabel("z, км", color=fg, labelpad=8)
    ax.set_zlabel("CAPE, Дж/кг", color=fg, labelpad=8)

    title = "CAPE(x, z) — 3D"
    if cape_max is not None and cape_mean is not None:
        title += f"   max={cape_max:.0f}  mean={cape_mean:.0f} Дж/кг"
    ax.set_title(title, color=fg, fontsize=11)

    ax.tick_params(colors=fg, labelsize=8)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor(bg)
        pane.pane.set_edgecolor(grid)

    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, color=grid, alpha=0.4)

    cb = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.10, aspect=18)
    cb.set_label("CAPE, Дж/кг", color=fg)
    cb.ax.tick_params(colors=fg)

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=bg)
    plt.close(fig)
    return buf.getvalue()

def make_storm2d_map(x_mc, z_mc, cape_2d, dark=False,
                     cape_max=None, cape_mean=None,
                     lfc_x=None, el_x=None, traffic=None):
    """
    Map of CAPE(x, z) as a pcolormesh.

    Parameters
    ----------
    x_mc : (nx,)        horizontal cell centers, m
    z_mc : (nz,)        vertical cell centers, m
    cape_2d : (nz, nx)  CAPE contribution field, J/kg per layer
    dark : bool
    cape_max, cape_mean : float or None
    lfc_x, el_x : (nx,) or None
    traffic : dict or None

    Returns
    -------
    bytes
        Raw PNG.
    """
    if dark:
        plt.style.use("dark_background")
        bg = "#0d1117"
        fg = "#e6edf3"
        grid = "#30363d"
    else:
        plt.style.use("default")
        bg = "#ffffff"
        fg = "#1f2328"
        grid = "#d0d7de"

    fig, ax = plt.subplots(figsize=(9.0, 5.2), facecolor=bg)
    ax.set_facecolor(bg)

    x_km = x_mc / 1000.0
    z_km = z_mc / 1000.0

    cmax = float(np.max(cape_2d)) if cape_2d.size > 0 else 1.0
    if cmax <= 0.0:
        cmax = 1.0

    im = ax.pcolormesh(x_km, z_km, cape_2d,
                       cmap="YlOrRd", vmin=0.0, vmax=cmax,
                       shading="nearest")

    # LFC / EL curves on top
    if lfc_x is not None and el_x is not None:
        mask_lfc = np.isfinite(lfc_x)
        mask_el = np.isfinite(el_x)
        if np.any(mask_lfc):
            ax.plot(x_km[mask_lfc], lfc_x[mask_lfc] / 1000.0,
                    color="#a371f7", linewidth=1.5, linestyle="--",
                    label="LFC")
        if np.any(mask_el):
            ax.plot(x_km[mask_el], el_x[mask_el] / 1000.0,
                    color="#f0883e", linewidth=1.5, linestyle=":",
                    label="EL")

    ax.set_xlabel("x, \u043a\u043c", fontsize=10, color=fg)
    ax.set_ylabel("z, \u043a\u043c", fontsize=10, color=fg)
    title = "\u041a\u0430\u0440\u0442\u0430 CAPE(x, z), \u0414\u0436/\u043a\u0433"
    ax.set_title(title, fontsize=11, color=fg)
    ax.tick_params(colors=fg, labelsize=9)
    for s in ax.spines.values():
        s.set_color(grid)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("CAPE, \u0414\u0436/\u043a\u0433", fontsize=9, color=fg)
    cbar.ax.tick_params(colors=fg, labelsize=8)

    if np.any(mask_lfc) or np.any(mask_el):
        ax.legend(loc="upper right", fontsize=8, facecolor=bg,
                  edgecolor=grid, labelcolor=fg)

    # Annotation bottom
    if cape_max is not None:
        info = f"max CAPE = {cape_max:.0f} \u0414\u0436/\u043a\u0433"
        if cape_mean is not None:
            info += f"   mean = {cape_mean:.0f} \u0414\u0436/\u043a\u0433"
        ax.text(0.5, -0.14, info, transform=ax.transAxes,
                ha="center", va="top", fontsize=9, color=fg)

    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=bg)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def make_storm2d_profile(x_mc, cape_col, dark=False, traffic=None):
    """
    Horizontal profile of total CAPE(x).
    """
    if dark:
        plt.style.use("dark_background")
        bg = "#0d1117"
        fg = "#e6edf3"
        grid = "#30363d"
        col = "#f0883e"
    else:
        plt.style.use("default")
        bg = "#ffffff"
        fg = "#1f2328"
        grid = "#d0d7de"
        col = "#bc4c00"

    fig, ax = plt.subplots(figsize=(9.0, 3.2), facecolor=bg)
    ax.set_facecolor(bg)

    x_km = x_mc / 1000.0
    ax.plot(x_km, cape_col, color=col, linewidth=2.0)
    ax.fill_between(x_km, 0.0, cape_col, color=col, alpha=0.20)

    # Thresholds
    ax.axhline(300.0, color="#f4a261", linestyle="--", linewidth=1.0,
               alpha=0.7, label="300 \u0414\u0436/\u043a\u0433 (\u043b\u0438\u0432\u0435\u043d\u044c)")
    ax.axhline(1000.0, color="#e63946", linestyle=":", linewidth=1.0,
               alpha=0.7, label="1000 \u0414\u0436/\u043a\u0433 (\u0433\u0440\u043e\u0437\u0430)")

    ax.set_xlabel("x, \u043a\u043c", fontsize=10, color=fg)
    ax.set_ylabel("CAPE, \u0414\u0436/\u043a\u0433", fontsize=10, color=fg)
    ax.set_title("\u0413\u043e\u0440\u0438\u0437\u043e\u043d\u0442\u0430\u043b\u044c\u043d\u044b\u0439 \u043f\u0440\u043e\u0444\u0438\u043b\u044c CAPE(x)",
                 fontsize=11, color=fg)
    ax.tick_params(colors=fg, labelsize=9)
    for s in ax.spines.values():
        s.set_color(grid)
    ax.grid(True, color=grid, alpha=0.4, linewidth=0.6)
    ax.legend(loc="upper right", fontsize=8, facecolor=bg,
              edgecolor=grid, labelcolor=fg)

    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=bg)
    plt.close(fig)
    buf.seek(0)
    return buf.read()
