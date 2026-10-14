"""Визуализация тумана: профиль qc(z) — радиационный, поле qc(x, z) — адвективный."""
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_snapshots_fog_1d(qc_hist, z_mc, times,
                          ncols=4, figsize=(16, 8), dark=False):
    """
    Сетка кадров: профиль qc(z) в г/кг.
    qc_hist : список массивов (nlev,)
    z_mc    : (nlev,)
    times   : список моментов
    """
    style = "dark_background" if dark else "default"
    line_color = "#00e5ff" if dark else "#00695c"

    with plt.style.context(style):
        n = len(times)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        fig.subplots_adjust(hspace=0.45, wspace=0.35)

        z_km = z_mc / 1000.0
        qc_all = [q * 1000.0 for q in qc_hist]   # г/кг
        qmax = max((q.max() for q in qc_all), default=0.0) or 1.0
        x_hi = max(qmax * 1.2, 0.05)

        for i in range(n):
            r, c = i // ncols, i % ncols
            ax = axes[r, c]
            ax.plot(qc_all[i], z_km, color=line_color, lw=1.6)
            ax.set_xlim(0, x_hi)
            ax.set_ylim(0, 20)
            ax.grid(True, alpha=0.3)
            ax.set_title(f"t = {times[i]:.0f} с  |  qc_surf = {qc_all[i][0]:.3f} г/кг",
                         fontsize=10)
            ax.tick_params(labelsize=8)
            if c == 0:
                ax.set_ylabel("z, км", fontsize=9)
            else:
                ax.set_yticklabels([])
            if r == nrows - 1:
                ax.set_xlabel("qc, г/кг", fontsize=9)
            else:
                ax.set_xticklabels([])

        for i in range(n, nrows * ncols):
            r, c = i // ncols, i % ncols
            axes[r, c].axis("off")

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=90, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")


def make_snapshots_fog_2d(qc_hist, x_mc, z_ifc, z_mc, times,
                          ncols=4, figsize=(17, 9), dark=False):
    """
    Сетка кадров: поле qc(x, z) (заливка, г/кг).
    qc_hist : список массивов (nz, nx)
    x_mc    : (nx,)
    z_ifc   : (nz+1,)
    z_mc    : (nz,)
    """
    style = "dark_background" if dark else "default"

    with plt.style.context(style):
        n = len(times)
        nrows = (n + ncols - 1) // ncols
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(
            nrows, ncols,
            left=0.06, right=0.98, top=0.94, bottom=0.10,
            wspace=0.15, hspace=0.30,
        )

        x_km = x_mc / 1000.0
        z_km_mc = z_mc / 1000.0   # qc живёт на центрах ячеек

        qmax = max((q.max() for q in qc_hist), default=0.0) or 1.0
        qmax_g = qmax * 1000.0

        im = None
        for i in range(n):
            r, c = i // ncols, i % ncols
            ax = fig.add_subplot(gs[r, c])
            qc_g = qc_hist[i] * 1000.0

            im = ax.pcolormesh(
                x_km, z_km_mc, qc_g,
                cmap="YlGnBu", vmin=0.0, vmax=qmax_g * 1.2,
                shading="nearest",
            )
            ax.set_ylim(0, 5)
            ax.set_xlim(x_km[0], x_km[-1])
            ax.set_title(f"t = {times[i]:.0f} с  |  max = {qc_g.max():.3f} г/кг",
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

        for i in range(n, nrows * ncols):
            r, c = i // ncols, i % ncols
            ax = fig.add_subplot(gs[r, c])
            ax.axis("off")

        if im is not None:
            cax = fig.add_axes([0.30, 0.04, 0.40, 0.020])
            cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
            cbar.set_label("qc, г/кг", fontsize=9)
            cbar.ax.tick_params(labelsize=8)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=95, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")