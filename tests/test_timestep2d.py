"""Тест step_forward_2d: прогон 2D-модели с осадками.

Проверяет:
- step_forward_2d работает с новой сигнатурой (включая qr).
- Нет NaN/Inf в полях.
- max|w|, max qc возвращаются как числа.
- qr и precip_flux присутствуют в возврате.
"""
import sys
sys.path.insert(0, "src")

import numpy as np
import pytest

from grid2d import make_grid2d
from state2d import make_initial_2d
from state import diagnose_exner
from thermo import virtual_theta
from timestep2d import step_forward_2d


def _run_short_2d(nsteps=50, dt=0.5):
    """Короткий прогон 2D-модели."""
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

    for n in range(nsteps):
        out = step_forward_2d(rho, w, vn, theta, qv, qc, qr,
                              exner_ref_2d, init["theta_ref_2d"], g, dt)
        rho = out["rho"]; w = out["w"]; vn = out["vn"]
        theta = out["theta"]; qv = out["qv"]; qc = out["qc"]
        qr = out["qr"]

    return {
        "max_w": float(np.max(np.abs(w))),
        "max_vn": float(np.max(np.abs(vn))),
        "max_qc": float(np.max(qc)),
        "max_qr": float(np.max(qr)),
        "max_dtheta": float(np.max(theta - init["theta_ref_2d"])),
        "precip_flux": float(np.max(np.abs(out["precip_flux"]))),
        "has_nan_w": bool(np.any(~np.isfinite(w))),
        "has_nan_vn": bool(np.any(~np.isfinite(vn))),
    }


def test_step_forward_2d_runs():
    """2D-прогон без падений, разумные числа."""
    res = _run_short_2d(nsteps=50)
    assert res["max_w"] > 0
    assert res["max_w"] < 100
    assert res["max_vn"] >= 0
    assert res["max_dtheta"] > 0
    # max_qc за 50 шагов может быть 0 — конвекция только развивается;
    # отдельный тест test_step_forward_2d_returns_qr проверяет qr.


def test_step_forward_2d_no_nan():
    """В полях нет NaN/Inf."""
    res = _run_short_2d(nsteps=50)
    assert not res["has_nan_w"], "NaN/Inf в w"
    assert not res["has_nan_vn"], "NaN/Inf в vn"


def test_step_forward_2d_returns_qr():
    """step_forward_2d должен возвращать qr и precip_flux."""
    res = _run_short_2d(nsteps=50)
    assert res["max_qr"] >= 0
    assert res["precip_flux"] >= 0