"""Тест step_forward: прогон 1D с турбулентностью и без.

Проверяет:
- step_forward работает без ошибок с новой сигнатурой (включая qr).
- max|w| возвращается как число.
- турбулентность не ломает расчёт.
"""
import sys
sys.path.insert(0, "src")

import numpy as np
import pytest

from grid import make_vertical_grid
from state import make_reference_atmosphere
from timestep import step_forward


def _run_short(nsteps=100, use_diffusion=True):
    """Короткий прогон 1D, возвращает финальное max|w| и qr_max."""
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
    qv = 0.020 * shape
    qc = np.zeros_like(theta)
    qr = np.zeros_like(theta)
    w = np.zeros(len(z_ifc))

    for n in range(nsteps):
        out = step_forward(rho, w, theta, qv, qc, qr,
                           exner_ref, theta_ref,
                           wgtfac_c, inv_dz, dz, z_mc, z_ifc, dt=0.5,
                           use_diffusion=use_diffusion)
        rho = out["rho"]; w = out["w"]
        theta = out["theta"]; qv = out["qv"]
        qc = out["qc"]; qr = out["qr"]

    return {
        "max_w": float(np.max(np.abs(w))),
        "max_qv": float(np.max(qv)),
        "max_qc": float(np.max(qc)),
        "max_qr": float(np.max(qr)),
        "max_dtheta": float(np.max(theta - theta_ref)),
        "precip_flux": float(out["precip_flux"]),
    }


def test_step_forward_no_diffusion():
    """Прогон без турбулентности — не падает, возвращает разумные числа."""
    res = _run_short(nsteps=100, use_diffusion=False)
    assert res["max_w"] > 0
    assert res["max_w"] < 100  # разумный предел
    assert res["max_qv"] > 0
    assert res["max_dtheta"] > 0


def test_step_forward_with_diffusion():
    """Прогон с турбулентностью — не падает, возвращает разумные числа."""
    res = _run_short(nsteps=100, use_diffusion=True)
    assert res["max_w"] > 0
    assert res["max_w"] < 100
    assert res["max_qv"] > 0
    assert res["max_dtheta"] > 0


def test_step_forward_returns_qr_and_precip():
    """step_forward должен возвращать qr и precip_flux."""
    res = _run_short(nsteps=50, use_diffusion=True)
    assert "max_qr" in res
    assert res["max_qr"] >= 0
    assert "precip_flux" in res
    assert res["precip_flux"] >= 0