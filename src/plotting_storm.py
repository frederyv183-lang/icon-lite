"""
Визуализация грозовой диагностики: профиль θ_v(z) с зоной CAPE.

- Синяя линия — θ_v среды
- Оранжевая штриховая — θ_v частицы
- Зелёная закраска — Зона CAPE (где θ_v_part > θ_v_env)
- Отмечены уровни LCL, LFC, EL
"""
import io
import base64

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def make_storm_profile_bytes(z_mc, theta_v_env, theta_v_part, cape_profile,
                       lfc_m=None, el_m=None, z_lcl_m=None,
                       z_trop_m=None, dark=False,
                       cape_Jkg=None, li_K=None, max_w=0.0,
                       traffic=None):
    """
    Профиль θ_v(z) с закрашенной зоной CAPE.

    Возвращает base64-строку PNG.
    """
    if dark:
        plt.style.use("dark_background")
        bg = "#0d1117"
        fg = "#e6edf3"
        grid = "#30363d"
        col_env = "#58a6ff"
        col_part = "#f0883e"
        col_cape = "#39d353"
    else:
        plt.style.use("default")
        bg = "#ffffff"
        fg = "#1f2328"
        grid = "#d0d7de"
        col_env = "#0969da"
        col_part = "#bc4c00"
        col_cape = "#1a7f37"

    fig, ax = plt.subplots(figsize=(5.5, 7.5), facecolor=bg)
    ax.set_facecolor(bg)

    z_km = z_mc / 1000.0

    # Зона CAPE: Закрашиваем между θ_v_env и θ_v_part там, где cape_profile > 0
    mask = cape_profile > 0.0
    if np.any(mask):
        idx = np.where(mask)[0]
        i0 = max(idx[0] - 1, 0)
        i1 = min(idx[-1] + 1, len(z_km) - 1)
        z_poly = np.concatenate([z_km[i0:i1 + 1],
                                 z_km[i0:i1 + 1][::-1]])
        x_poly = np.concatenate([theta_v_part[i0:i1 + 1],
                                 theta_v_env[i0:i1 + 1][::-1]])
        poly = Polygon(np.column_stack([x_poly, z_poly]),
                       closed=True, facecolor=col_cape, alpha=0.25,
                       edgecolor=col_cape, linewidth=1.2,
                       label="CAPE")
        ax.add_patch(poly)

    ax.plot(theta_v_env, z_km, color=col_env, linewidth=2.0,
            label=r"Среда $\theta_v$")
    ax.plot(theta_v_part, z_km, color=col_part, linewidth=2.0,
            linestyle="--", label=r"Частица $\theta_v$")

    xlo = float(min(theta_v_env.min(), theta_v_part.min()))
    xhi = float(max(theta_v_env.max(), theta_v_part.max()))
    xpad = (xhi - xlo) * 0.05
    xlo -= xpad
    xhi += xpad
    ax.set_xlim(xlo, xhi)

    # Уровни
    if lfc_m is not None:
        ax.axhline(lfc_m / 1000.0, color=col_cape, linestyle=":",
                   linewidth=1.2, alpha=0.9)
        ax.text(xlo + xpad, lfc_m / 1000.0 + 0.15,
                f"LFC {lfc_m:.0f} м", color=col_cape, fontsize=8)
    if el_m is not None:
        ax.axhline(el_m / 1000.0, color=col_part, linestyle=":",
                   linewidth=1.2, alpha=0.9)
        ax.text(xlo + xpad, el_m / 1000.0 + 0.15,
                f"EL {el_m:.0f} м", color=col_part, fontsize=8)
    if z_lcl_m is not None and z_lcl_m > 0:
        ax.axhline(z_lcl_m / 1000.0, color="#a371f7", linestyle="-.",
                   linewidth=1.0, alpha=0.8)
        ax.text(xlo + xpad, z_lcl_m / 1000.0 + 0.15,
                f"LCL {z_lcl_m:.0f} м", color="#a371f7", fontsize=8)

    ax.set_xlabel(r"$\theta_v$, K", fontsize=10, color=fg)
    ax.set_ylabel("z, км", fontsize=10, color=fg)
    ax.set_title("Профиль виртуальной потенциальной температуры",
                 fontsize=11, color=fg)
    ax.tick_params(colors=fg, labelsize=9)
    for s in ax.spines.values():
        s.set_color(grid)
    ax.grid(True, color=grid, alpha=0.4, linewidth=0.6)
    ax.legend(loc="upper right", fontsize=9, facecolor=bg,
              edgecolor=grid, labelcolor=fg)

    # Подпись с числами внизу
    if cape_Jkg is not None and li_K is not None:
        info = f"CAPE = {cape_Jkg:.0f} ж/кг   LI = {li_K:.1f} K   max|w| = {max_w:.1f} м/с"
        ax.text(0.5, -0.08, info, transform=ax.transAxes,
                ha="center", va="top", fontsize=9, color=fg)

    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=bg)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def make_storm_profile(*args, **kwargs):
    """Backward compat: returns base64-encoded PNG (ascii str)."""
    png_bytes = make_storm_profile_bytes(*args, **kwargs)
    return base64.b64encode(png_bytes).decode("ascii")



def make_storm_timeseries(times, w_max_hist, dark=False,
                          cape_Jkg=None, li_K=None, max_w=None):
    """
    Timeseries plot of max|w|(t).

    Parameters
    ----------
    times : list[float]
        Time points, seconds.
    w_max_hist : list[float]
        max|w| at each time point, m/s.
    dark : bool
        Use dark background.
    cape_Jkg, li_K, max_w : float or None
        Optional annotations.

    Returns
    -------
    bytes
        Raw PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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

    fig, ax = plt.subplots(figsize=(6.5, 3.8), facecolor=bg)
    ax.set_facecolor(bg)

    t = list(times)
    w = list(w_max_hist)
    if len(t) > 0 and len(w) == len(t):
        ax.plot(t, w, color=col, linewidth=2.0, marker="o",
                markersize=4, label=r"max$|w|$")

    # Reference levels
    ax.axhline(15.0, color="#e63946", linestyle="--", linewidth=1.0,
               alpha=0.7, label="15 м/с (гроза)")
    ax.axhline(25.0, color="#d62828", linestyle=":", linewidth=1.0,
               alpha=0.7, label="25 м/с (сильная)")

    ax.set_xlabel("t, с", fontsize=10, color=fg)
    ax.set_ylabel(r"max$|w|$, м/с", fontsize=10, color=fg)
    ax.set_title("Эволюция вертикальной скорости", fontsize=11, color=fg)
    ax.tick_params(colors=fg, labelsize=9)
    for s in ax.spines.values():
        s.set_color(grid)
    ax.grid(True, color=grid, alpha=0.4, linewidth=0.6)
    ax.legend(loc="upper left", fontsize=8, facecolor=bg,
              edgecolor=grid, labelcolor=fg)

    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=bg)
    plt.close(fig)
    buf.seek(0)
    return buf.read()
