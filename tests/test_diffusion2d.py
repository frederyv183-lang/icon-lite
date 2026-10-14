"""Юнит-тесты для src/diffusion2d.py."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from diffusion2d import (deformation_2d, richardson_number_2d,
                         km_smagorinsky_2d, diffusion_step_2d)


def _make_grid(nz, nx, dz_val=500.0, dx_val=500.0):
    dz = np.full(nz, dz_val)
    dx = np.full(nx, dx_val)
    dz_ifc = np.zeros(nz + 1)
    dz_ifc[1:-1] = 0.5 * (dz[:-1] + dz[1:])
    dz_ifc[0] = dz[0]
    dz_ifc[-1] = dz[-1]
    dx_ifc = np.zeros(nx + 1)
    dx_ifc[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    dx_ifc[0] = dx[0]
    dx_ifc[-1] = dx[-1]
    inv_dz = 1.0 / dz
    return dz, dz_ifc, dx, dx_ifc, inv_dz


def test_deformation_zero():
    """При w=0, vn=0 скорость деформации близка к 0."""
    nz, nx = 20, 30
    w = np.zeros((nz + 1, nx))
    vn = np.zeros((nz, nx + 1))
    dz, dz_ifc, dx, dx_ifc, inv_dz = _make_grid(nz, nx)
    S = deformation_2d(w, vn, inv_dz, dz_ifc, dx, dx_ifc)
    assert S.shape == (nz, nx)
    assert np.all(S < 1e-5)


def test_km_shape_and_nonneg():
    """Km — (nz, nx), неотрицательно, без NaN."""
    nz, nx = 20, 30
    np.random.seed(42)
    theta_v = 300 + 5 * np.random.randn(nz, nx)
    w = 0.01 * np.random.randn(nz + 1, nx)
    vn = 0.01 * np.random.randn(nz, nx + 1)
    dz, dz_ifc, dx, dx_ifc, inv_dz = _make_grid(nz, nx)
    Km = km_smagorinsky_2d(theta_v, w, vn, inv_dz, dz, dz_ifc, dx, dx_ifc)
    assert Km.shape == (nz, nx)
    assert np.all(Km >= 0.0)
    assert not np.isnan(Km).any()


def test_diffusion_constant_field():
    """Диффузия константы не меняет поле."""
    nz, nx = 10, 15
    field = np.full((nz, nx), 300.0)
    Km = np.full((nz, nx), 10.0)
    dz, dz_ifc, dx, dx_ifc, _ = _make_grid(nz, nx)
    new = diffusion_step_2d(field, Km, dz, dz_ifc, dx, dx_ifc, 1.0)
    assert np.allclose(new, field)


def test_diffusion_smooths_peak():
    """Горка сглаживается: пик уменьшается, соседи получают часть.

    За один шаг диффузия распространяется на ~sqrt(Km*dt), что меньше
    размера ячейки при dt=0.5, Km=100, dz=dx=100. Поэтому проверяем
    ближайшего соседа, а не угол домена.
    """
    nz, nx = 11, 11
    field = np.zeros((nz, nx))
    field[nz // 2, nx // 2] = 100.0
    Km = np.full((nz, nx), 100.0)
    dz, dz_ifc, dx, dx_ifc, _ = _make_grid(nz, nx, dz_val=100.0, dx_val=100.0)

    sum_before = field.sum()
    new = diffusion_step_2d(field, Km, dz, dz_ifc, dx, dx_ifc, 0.5)

    # Пик уменьшился
    assert new[nz // 2, nx // 2] < field[nz // 2, nx // 2]
    # Сосед получил часть
    assert new[nz // 2, nx // 2 + 1] > 0.0
    assert new[nz // 2 + 1, nx // 2] > 0.0
    # Сумма сохранилась (диффузия с нулевыми потоками на границах)
    assert abs(new.sum() - sum_before) < 1e-6 * sum_before


def test_diffusion_stable_no_nan():
    """50 шагов диффузии без NaN и без разлёта."""
    nz, nx = 20, 30
    np.random.seed(0)
    field = 300 + np.random.randn(nz, nx)
    Km = np.abs(np.random.randn(nz, nx)) * 5.0
    dz, dz_ifc, dx, dx_ifc, _ = _make_grid(nz, nx)
    for _ in range(50):
        field = diffusion_step_2d(field, Km, dz, dz_ifc, dx, dx_ifc, 0.1)
    assert not np.isnan(field).any()
    assert np.max(np.abs(field - 300)) < 100.0