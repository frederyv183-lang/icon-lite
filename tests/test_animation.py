"""Тест визуализации 1D: make_snapshots_frame возвращает валидный PNG."""
import sys
sys.path.insert(0, "src")

import base64
import numpy as np
import pytest

from grid import make_vertical_grid
from state import make_reference_atmosphere
from timestep import step_forward
from animation import make_snapshots_frame


def _run_short_1d(nsteps=100, n_snap=4):
    """Короткий прогон 1D-модели, возвращает историю для визуализации."""
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

    snap_every = max(1, nsteps // n_snap)
    times = [0.0]
    w_hist = [w.copy()]
    qv_hist = [qv.copy()]
    qc_hist = [qc.copy()]

    dt = 0.5
    for n in range(nsteps):
        out = step_forward(rho, w, theta, qv, qc, exner_ref, theta_ref,
                           wgtfac_c, inv_dz, dz, z_mc, z_ifc, dt,
                           use_diffusion=True, dx=500.0)
        rho, w = out["rho"], out["w"]
        theta, qv, qc = out["theta"], out["qv"], out["qc"]
        if (n + 1) % snap_every == 0 and len(times) < n_snap:
            times.append((n + 1) * dt)
            w_hist.append(w.copy())
            qv_hist.append(qv.copy())
            qc_hist.append(qc.copy())

    return w_hist, qv_hist, qc_hist, z_ifc, z_mc, times


def test_animation_returns_valid_png():
    """make_snapshots_frame должен вернуть base64-строку с PNG внутри."""
    w_hist, qv_hist, qc_hist, z_ifc, z_mc, times = _run_short_1d()

    b64 = make_snapshots_frame(w_hist, qv_hist, qc_hist,
                                z_ifc, z_mc, times)

    # это строка
    assert isinstance(b64, str)
    assert len(b64) > 100

    # декодируется в байты
    png_bytes = base64.b64decode(b64)

    # PNG-сигнатура: 89 50 4E 47 0D 0A 1A 0A
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png_bytes) > 1000


def test_animation_history_lengths_match():
    """Длины историй w, qv, qc и times должны совпадать."""
    w_hist, qv_hist, qc_hist, _, _, times = _run_short_1d()

    assert len(w_hist) == len(times)
    assert len(qv_hist) == len(times)
    assert len(qc_hist) == len(times)
    assert len(times) >= 2