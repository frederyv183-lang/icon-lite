"""
2D-визуализация: контурные поля w(x, z) и qc(x, z).
Цветовая шкала — снизу, отдельно от графиков.
"""
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_snapshots_2d(w_hist, qc_hist, qr_hist, x_mc, z_ifc, z_mc, times,
                      ncols=4, figsize=(17, 9),
                      w_floor=1.0, qc_levels=(0.5, 1.0, 2.0),
                      qr_levels=(0.05, 0.1, 0.2), dark=False):
    """
    Сетка кадров: поле w(x,z) + контуры qc(x,z).
    """
    n = len(times)
    nrows = (n + ncols - 1) // ncols

    # Выбор стиля в зависимости от темы
    style = "dark_background" if dark else "default"
    grid_color = "#00e5ff" if dark else "k"
    zero_color = "#00e5ff" if dark else "k"
    qr_contour_color = "#7dd3fc" if dark else "blue"

    with plt.style.context(style):
        return _draw_snapshots_2d(
            w_hist, qc_hist, qr_hist, x_mc, z_ifc, z_mc, times,
            ncols, figsize, w_floor, qc_levels, qr_levels,
            dark, zero_color, qr_contour_color,
        )


# ---------------------------------------------------------------------
#  3D-снимки: сетка кадров w(x, z) через plot_surface
# ---------------------------------------------------------------------

def make_snapshots_2d_3d(w_hist, qc_hist, qr_hist, x_mc, z_ifc, z_mc, times,
                         ncols=4, figsize=(18, 10),
                         w_floor=1.0, dark=False,
                         elev=26.0, azim=-60.0):
    """
    3D-сетка кадров: plot_surface(w(x,z)) с контурами qc на полу.

    Возвращает base64-строку PNG (как и make_snapshots_2d).
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    n = len(times)
    nrows = (n + ncols - 1) // ncols

    style = "dark_background" if dark else "default"
    with plt.style.context(style):
        return _draw_snapshots_2d_3d(
            w_hist, qc_hist, qr_hist, x_mc, z_ifc, z_mc, times,
            ncols, figsize, w_floor, nrows,
            dark, elev, azim,
        )


def _draw_snapshots_2d_3d(w_hist, qc_hist, qr_hist, x_mc, z_ifc, z_mc, times,
                          ncols, figsize, w_floor, nrows,
                          dark, elev, azim):
    bg = "#0d1117" if dark else "#ffffff"
    fg = "#e6edf3" if dark else "#1f2328"
    grid = "#30363d" if dark else "#d0d7de"

    x_km = x_mc / 1000.0
    z_km = z_mc / 1000.0
    X, Z = np.meshgrid(x_km, z_km)  # (nz, nx)

    w_max_global = max(np.max(np.abs(wh)) for wh in w_hist) or 1.0

    fig = plt.figure(figsize=figsize, facecolor=bg)

    for i in range(len(times)):
        r = i // ncols
        c = i % ncols
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        ax.set_facecolor(bg)

        w = w_hist[i]
        w_mc = 0.5 * (w[:-1] + w[1:])  # (nz+1, nx) -> (nz, nx)

        ax.plot_surface(
            X, Z, w_mc,
            cmap="RdBu_r", vmin=-w_max_global, vmax=w_max_global,
            linewidth=0, antialiased=True, alpha=0.95,
            rstride=1, cstride=1,
        )

        qc = qc_hist[i]
        if np.max(qc) > 0:
            ax.contour(
                X, Z, qc,
                levels=[0.5e-3, 1.0e-3, 2.0e-3],
                colors=["#7dd3fc"] if dark else ["#2563eb"],
                linewidths=1.0, offset=0.0,
            )

        ax.set_xlabel("x, км", color=fg, fontsize=7, labelpad=2)
        ax.set_ylabel("z, км", color=fg, fontsize=7, labelpad=2)
        ax.set_zlabel("w, м/с", color=fg, fontsize=7, labelpad=2)
        ax.set_title(f"t = {times[i]:.0f} с", color=fg, fontsize=9)

        ax.tick_params(colors=fg, labelsize=6)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.set_facecolor(bg)
            pane.pane.set_edgecolor(grid)

        ax.view_init(elev=elev, azim=azim)
        ax.grid(True, color=grid, alpha=0.3)

    fig.subplots_adjust(left=0.04, right=0.94, top=0.94, bottom=0.06,
                        wspace=0.05, hspace=0.10)

    cax = fig.add_axes([0.955, 0.15, 0.012, 0.70])
    sm = plt.cm.ScalarMappable(
        cmap="RdBu_r",
        norm=plt.Normalize(vmin=-w_max_global, vmax=w_max_global),
    )
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("w, м/с", color=fg)
    cb.ax.tick_params(colors=fg)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=105, facecolor=bg)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def _draw_snapshots_2d(w_hist, qc_hist, qr_hist, x_mc, z_ifc, z_mc, times,
                       ncols, figsize, w_floor, qc_levels, qr_levels,
                       dark, zero_color, qr_contour_color):
    """Внутренняя функция — рисует 2D-снимки в текущем стиле."""
    n = len(times)
    nrows = (n + ncols - 1) // ncols

    # Создаём фигуру + отдельный нижний Axes для colorbar
    fig = plt.figure(figsize=figsize)
    # Сетка графиков: оставляем 8% внизу под colorbar
    gs = fig.add_gridspec(
        nrows, ncols,
        left=0.06, right=0.98, top=0.94, bottom=0.10,
        wspace=0.15, hspace=0.30,
    )

    x_km = x_mc / 1000.0
    z_km_ifc = z_ifc / 1000.0
    z_km_mc = z_mc / 1000.0

    w_max_global = max(np.max(np.abs(wh)) for wh in w_hist) or 1.0

    im_w = None
    axes = []

    for i in range(n):
        r = i // ncols
        c = i % ncols
        ax = fig.add_subplot(gs[r, c])
        axes.append(ax)

        w_field = w_hist[i]
        w_abs = np.max(np.abs(w_field))
        w_scale = max(w_abs, w_floor)

        im_w = ax.pcolormesh(
            x_km, z_km_ifc, w_field,
            cmap="RdBu_r", vmin=-w_scale, vmax=w_scale,
            shading="nearest",
        )

        # контуры qc
        qc_field = qc_hist[i] * 1000
        if qc_field.max() > 0.1:
            levels = [l for l in qc_levels if l < qc_field.max()]
            if levels:
                cs = ax.contour(
                    x_km, z_km_mc, qc_field, levels=levels,
                    colors="lime", linewidths=1.8,
                )
                ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")

        # контуры qr (пунктирные, синие)
        qr_field = qr_hist[i] * 1000
        if qr_field.max() > 0.01:
            qr_lev = [l for l in qr_levels if l < qr_field.max()]
            if qr_lev:
                cs_r = ax.contour(
                    x_km, z_km_mc, qr_field, levels=qr_lev,
                    colors=qr_contour_color, linewidths=1.4, linestyles="dashed",
                )
                ax.clabel(cs_r, inline=True, fontsize=6, fmt="%.2f")

        ax.set_ylim(0, 15)
        ax.set_xlim(x_km[0], x_km[-1])
        ax.set_title(f"t = {times[i]:.0f} с | w_max = {w_abs:.1f} м/с",
                     fontsize=10)
        ax.tick_params(labelsize=8)
        if c == 0:
            ax.set_ylabel("z, км", fontsize=9)
        else:
            ax.set_yticklabels([])
        if r == nrows - 1:
            ax.set_xlabel("x, км", fontsize=9)
        else:
            ax.set_xticklabels([])

    # убираем пустые ячейки
    for i in range(n, nrows * ncols):
        r = i // ncols
        c = i % ncols
        ax = fig.add_subplot(gs[r, c])
        ax.axis("off")

    # Colorbar — на отдельной оси внизу
    if im_w is not None:
        cax = fig.add_axes([0.30, 0.04, 0.40, 0.020])  # [left, bottom, width, height]
        cbar = fig.colorbar(im_w, cax=cax, orientation="horizontal")
        cbar.set_label("w, м/с (своя шкала для каждого кадра)", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle("Эволюция конвективного облака (2D)",
                 fontsize=13, y=0.985)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=95, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")