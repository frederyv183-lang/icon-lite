"""Генерация набора кадров для показа эволюции модели (1D)."""
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_snapshots_frame(w_hist, qv_hist, qc_hist, qr_hist,
                         z_ifc, z_mc, times,
                         ncols=4, figsize=(16, 8)):
    """
    Рисует сетку кадров nrows x ncols: для каждого снимка три профиля
    (w, qv, qc) и опционально qr (красная линия).

    Параметры
    ---------
    w_hist  : список массивов (nlev+1,) — вертикальная скорость
    qv_hist : список массивов (nlev,) — водяной пар
    qc_hist : список массивов (nlev,) — облачная вода
    qr_hist : список массивов (nlev,) — дождевая вода
    z_ifc   : (nlev+1,) высоты интерфейсов (для w)
    z_mc    : (nlev,)  высоты центров (для qv, qc, qr)
    times   : список моментов времени, с

    Возвращает base64-строку PNG.
    """
    n = len(times)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=figsize,
                             squeeze=False)
    fig.subplots_adjust(hspace=0.4, wspace=0.35)

    # Для qr: если максимум меньше 1e-6, не рисуем (нет дождя)
    qr_max_global = max(np.max(qrh) for qrh in qr_hist) if qr_hist else 0.0
    has_qr = qr_max_global > 1e-6

    for i in range(n):
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]

        # Основной профиль: w (синий)
        ax.plot(w_hist[i], z_ifc / 1000.0, color="tab:blue", lw=1.2,
                label="w")

        # qr на той же оси, но с отдельной шкалой? Нет — тогда qc/qr не поместится.
        # Используем twin axis только для qr, если он есть.
        if has_qr:
            ax2 = ax.twiny()
            ax2.plot(qr_hist[i] * 1000, z_mc / 1000.0,
                     color="tab:red", lw=1.4, label="qr, г/кг")
            ax2.set_xlim(-0.01, max(0.5, qr_max_global * 1000 * 1.2))
            ax2.set_xlabel("qr, г/кг", fontsize=8, color="tab:red")
            ax2.tick_params(axis='x', labelsize=7, colors="tab:red")
            # убираем лишние спайны
            ax2.spines["top"].set_color("tab:red")
            ax2.spines["bottom"].set_visible(False)
            ax2.spines["left"].set_visible(False)
            ax2.spines["right"].set_visible(False)
        else:
            ax2 = None

        ax.set_xlim(-8.5, 8.5)
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