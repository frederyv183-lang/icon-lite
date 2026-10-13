"""Тест визуализации 2D: make_snapshots_2d возвращает валидный PNG."""
import sys
sys.path.insert(0, "src")

import base64
import numpy as np
import pytest

from grid2d import make_grid2d
from state2d import make_initial_2d
from state import diagnose_exner
from thermo import virtual_theta
from timestep2d import step_forward_2d
from plotting2d import make_snapshots_2d


def _run_short_2d(nsteps=30, dt=0.5, n_snap=4):
    """Короткий 2D-прогон с историей w, qc, qr."""
    g = make_grid2d(nz=40, nx=80, x_total=40000.0, z_top=20000.0)
    init = make_initial_2d(g, None, qv_max=0.020, dtheta=2.0,
                            x0_km=20.0, z0_km=3.0,
                            radius_x_km=5.0, radius_z_km=1.5)

    theta_v_ref = virtual_theta(init["theta_ref_2d"], np.zeros_like(init["qv"]))
    exner_ref_2d = diagnose_exner(init["rho"], theta_v_ref)

    rho = init["rho"].copy()
    w = init["w"].copy()
    vn = init["vn"].copy()
    theta = init["theta"].copy()
    qv = init["qv"].copy()
    qc = init["qc"].copy()
    qr = np.zeros_like(qc)

    snap_every = max(1, nsteps // n_snap)
    times = [0.0]
    w_hist = [w.copy()]
    qc_hist = [qc.copy()]
    qr_hist = [qr.copy()]

    for n in range(nsteps):
        out = step_forward_2d(rho, w, vn, theta, qv, qc, qr,
                              exner_ref_2d, init["theta_ref_2d"], g, dt)
        rho = out["rho"]; w = out["w"]; vn = out["vn"]
        theta = out["theta"]; qv = out["qv"]; qc = out["qc"]
        qr = out["qr"]
        if (n + 1) % snap_every == 0 and len(times) < n_snap:
            times.append((n + 1) * dt)
            w_hist.append(w.copy())
            qc_hist.append(qc.copy())
            qr_hist.append(qr.copy())

    return w_hist, qc_hist, qr_hist, g["x_mc"], g["z_ifc"], g["z_mc"], times


def test_plot2d_returns_valid_png():
    """make_snapshots_2d должен вернуть base64 PNG."""
    w_hist, qc_hist, qr_hist, x_mc, z_ifc, z_mc, times = _run_short_2d()

    b64 = make_snapshots_2d(w_hist, qc_hist, qr_hist,
                             x_mc, z_ifc, z_mc, times)

    assert isinstance(b64, str)
    assert len(b64) > 100

    png_bytes = base64.b64decode(b64)
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png_bytes) > 1000


def test_plot2d_history_lengths_match():
    """Длины историй w, qc, qr и times совпадают."""
    w_hist, qc_hist, qr_hist, _, _, _, times = _run_short_2d()

    assert len(w_hist) == len(times)
    assert len(qc_hist) == len(times)
    assert len(qr_hist) == len(times)
    assert len(times) >= 2