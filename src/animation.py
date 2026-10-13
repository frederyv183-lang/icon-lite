"""Генерация набора кадров для показа эволюции модели."""
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_snapshots_frame(w_hist, rho_hist, pi_hist,
                         z_ifc, z_mc, times,
                         ncols=4, figsize=(16, 8)):
    """
    Рисует сетку кадров nrows x ncols: для каждого снимка три профиля
    (w, rho, pi') на одной колонке кадра.
    Возвращает base64-строку PNG.
    """
    n = len(times)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=figsize,
                             squeeze=False)
    fig.subplots_adjust(hspace=0.4, wspace=0.35)

    for i in range(n):
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]
        ax.plot(w_hist[i], z_ifc / 1000.0, color="tab:blue", lw=1.2)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(0, 20)
        ax.axvline(0, color="k", lw=0.4, alpha=0.4)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"t = {times[i]:.0f} с", fontsize=10)
        if c == 0:
            ax.set_ylabel("z, км", fontsize=9)
        else:
            ax.set_yticklabels([])
        if r == nrows - 1:
            ax.set_xlabel("w, м/с", fontsize=9)
        ax.tick_params(labelsize=8)

    # убрать пустые оси, если n не кратно ncols
    for i in range(n, nrows * ncols):
        r = i // ncols
        c = i % ncols
        axes[r, c].axis("off")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")