"""График профиля температуры T(z) в °C для набора кадров."""
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diagnostics import diagnose_temperature, temperature_celsius


def make_snapshots_temperature(theta_hist, exner_hist, z_mc, times,
                               ncols=4, figsize=(16, 8), dark=False):
    """
    Рисует сетку кадров nrows x ncols: профиль T(z) в °C.

    Параметры
    ---------
    theta_hist : список массивов (nlev,) — потенциальная температура
    exner_hist : список массивов (nlev,) — Exner-функция
    z_mc       : (nlev,) — высоты центров, м
    times      : список моментов времени, с

    Возвращает base64-строку PNG.
    """
    style = "dark_background" if dark else "default"
    line_color = "#00e5ff" if dark else "#c0392b"

    with plt.style.context(style):
        return _draw(theta_hist, exner_hist, z_mc, times,
                     ncols, figsize, dark, line_color)


def _draw(theta_hist, exner_hist, z_mc, times,
          ncols, figsize, dark, line_color):
    n = len(times)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    fig.subplots_adjust(hspace=0.45, wspace=0.3)

    z_km = z_mc / 1000.0

    # Общий диапазон X по всем кадрам
    T_all = []
    for th, ex in zip(theta_hist, exner_hist):
        T_all.append(temperature_celsius(diagnose_temperature(th, ex)))
    T_min = min(T.min() for T in T_all)
    T_max = max(T.max() for T in T_all)
    pad = 5.0
    x_lo, x_hi = T_min - pad, T_max + pad

    for i in range(n):
        r, c = i // ncols, i % ncols
        ax = axes[r, c]

        T_C = temperature_celsius(diagnose_temperature(
            theta_hist[i], exner_hist[i]))

        ax.plot(T_C, z_km, color=line_color, lw=1.6)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(0, 20)
        ax.grid(True, alpha=0.3)

        T_surf = T_C[0]
        ax.set_title(f"t = {times[i]:.0f} с  |  T_surf = {T_surf:.1f} °C",
                     fontsize=10)
        ax.tick_params(labelsize=8)

        if c == 0:
            ax.set_ylabel("z, км", fontsize=9)
        else:
            ax.set_yticklabels([])
        if r == nrows - 1:
            ax.set_xlabel("T, °C", fontsize=9)
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