"""Визуализация результатов модели."""
import matplotlib
matplotlib.use("Agg")  # headless-режим (для сервера)
import matplotlib.pyplot as plt
import numpy as np


def plot_profiles(z_ifc, z_mc, w_history, rho_history, pi_history, times,
                  savepath="output.png"):
    """
    Рисует 3 панели: w(z), rho(z), pi'(z) в разные моменты времени.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    n_snapshots = len(times)
    colors = plt.cm.viridis(np.linspace(0, 1, n_snapshots))

    # --- w(z) ---
    ax = axes[0]
    for i, t in enumerate(times):
        ax.plot(w_history[i], z_ifc / 1000.0, color=colors[i],
                label=f"t = {t:.0f} с")
    ax.set_xlabel("w, м/с")
    ax.set_ylabel("z, км")
    ax.set_title("Вертикальная скорость")
    ax.axvline(0, color="k", lw=0.5, alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # --- rho(z) ---
    ax = axes[1]
    for i, t in enumerate(times):
        ax.plot(rho_history[i], z_mc / 1000.0, color=colors[i])
    ax.set_xlabel("ρ, кг/м³")
    ax.set_ylabel("z, км")
    ax.set_title("Плотность")
    ax.grid(True, alpha=0.3)

    # --- pi'(z) ---
    ax = axes[2]
    for i, t in enumerate(times):
        ax.plot(pi_history[i] * 1000, z_mc / 1000.0, color=colors[i])
    ax.set_xlabel("π' × 1000")
    ax.set_ylabel("z, км")
    ax.set_title("Возмущение Exner")
    ax.axvline(0, color="k", lw=0.5, alpha=0.3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(savepath, dpi=100, bbox_inches="tight")
    plt.close()
    return savepath
