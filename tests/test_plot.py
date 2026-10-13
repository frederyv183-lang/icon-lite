"""Тест визуализации 1D: plot_profiles сохраняет валидный PNG-файл."""
import sys
sys.path.insert(0, "src")

import os
import tempfile
import numpy as np
import pytest

from grid import make_vertical_grid
from state import make_reference_atmosphere
from timestep import step_forward
from plotting import plot_profiles


def _run_short_1d_with_history(nsteps=100, n_snap=4):
    """Короткий прогон 1D с сохранением истории w, rho, pi'."""
    grid = make_vertical_grid(nlev=40, z_top=20000.0, stretch=1.15)
    z_mc = grid["z_mc"]
    z_ifc = grid["z_ifc"]
    wgtfac_c = grid["wgtfac_c"]
    inv_dz = 1.0 / grid["dz"]
    dz = grid["dz"]

    ref = make_reference_atmosphere(z_mc, z_ifc, theta_surf=300.0)
    rho = ref["rho_ref"].copy()
    theta_ref = ref["theta_ref"]
    exner_ref = ref["exner_ref"]

    shape = np.exp(-((z_mc - 5000.0) / 1500.0) ** 2)
    theta = theta_ref + 2.0 * shape
    qv = 0.010 * shape
    qc = np.zeros_like(theta)
    w = np.zeros(len(z_ifc))
    pi_pr = np.zeros_like(rho)

    snap_every = max(1, nsteps // n_snap)
    times = [0.0]
    w_hist = [w.copy()]
    rho_hist = [rho.copy()]
    pi_hist = [pi_pr.copy()]

    dt = 0.5
    for n in range(nsteps):
        out = step_forward(rho, w, theta, qv, qc, exner_ref, theta_ref,
                           wgtfac_c, inv_dz, dz, z_mc, z_ifc, dt,
                           use_diffusion=True, dx=500.0)
        rho, w = out["rho"], out["w"]
        theta, qv, qc = out["theta"], out["qv"], out["qc"]
        pi_pr = out["pi_pr"]
        if (n + 1) % snap_every == 0 and len(times) < n_snap:
            times.append((n + 1) * dt)
            w_hist.append(w.copy())
            rho_hist.append(rho.copy())
            pi_hist.append(pi_pr.copy())

    return z_ifc, z_mc, w_hist, rho_hist, pi_hist, times


def test_plot_profiles_saves_png():
    """plot_profiles должен создать непустой PNG-файл."""
    z_ifc, z_mc, w_hist, rho_hist, pi_hist, times = _run_short_1d_with_history()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "output.png")
        result = plot_profiles(z_ifc, z_mc, w_hist, rho_hist, pi_hist,
                                times, savepath=out_path)

        assert os.path.exists(result)
        assert os.path.getsize(result) > 1000

        with open(result, "rb") as f:
            head = f.read(8)
        # PNG-сигнатура
        assert head == b"\x89PNG\r\n\x1a\n"


def test_plot_history_consistency():
    """Все истории должны иметь одинаковую длину, равную len(times)."""
    _, _, w_hist, rho_hist, pi_hist, times = _run_short_1d_with_history()

    assert len(w_hist) == len(times)
    assert len(rho_hist) == len(times)
    assert len(pi_hist) == len(times)
    assert len(times) >= 2