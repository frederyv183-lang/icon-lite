"""Юнит-тесты для src/diagnostics.py."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from diagnostics import (diagnose_temperature, temperature_celsius,
                         extrapolate_t_2m, extract_t_metrics)


def test_diagnose_temperature_basic():
    """T = theta * Pi."""
    theta = np.array([300.0, 310.0, 320.0])
    exner = np.array([1.0, 0.9, 0.8])
    T = diagnose_temperature(theta, exner)
    assert np.allclose(T, [300.0, 279.0, 256.0])


def test_temperature_celsius_shift():
    """Перевод K -> °C — сдвиг на 273.15."""
    T_K = np.array([273.15, 300.0, 373.15])
    T_C = temperature_celsius(T_K)
    assert np.allclose(T_C, [0.0, 26.85, 100.0])


def test_extrapolate_t_2m_linear():
    """Экстраполяция к 2 м при линейном профиле."""
    # T падает с высотой: 300 K на 100 м, 290 K на 600 м
    T_K = np.array([300.0, 290.0, 280.0])
    z_mc = np.array([100.0, 600.0, 1100.0])
    # slope = -10/500 = -0.02 K/м
    # T_2m = 300 + (-0.02)*(2 - 100) = 300 + 1.96 = 301.96
    T_2m = extrapolate_t_2m(T_K, z_mc, z_target=2.0)
    assert abs(T_2m - 301.96) < 1e-6


def test_extrapolate_t_2m_degenerate():
    """Если z_target >= z[0] — возвращаем T[0]."""
    T_K = np.array([300.0, 290.0])
    z_mc = np.array([100.0, 600.0])
    T_2m = extrapolate_t_2m(T_K, z_mc, z_target=200.0)
    assert T_2m == 300.0


def test_extract_t_metrics_keys():
    """Все ключи присутствуют, значения реалистичны."""
    nz = 20
    z_mc = np.linspace(100.0, 15000.0, nz)
    theta = 300.0 + 5.0 * np.exp(-((z_mc - 5000.0) / 1500.0) ** 2)
    exner = np.linspace(1.0, 0.4, nz)
    qc = np.zeros(nz)
    qc[8] = 5e-4  # облако на 8-м уровне

    m = extract_t_metrics(theta, exner, z_mc, qc)
    assert set(m.keys()) == {"T_surface_C", "T_2m_C", "T_cloud_C",
                             "T_min_C", "T_max_C"}
    assert m["T_surface_C"] == pytest.approx(26.85, abs=0.1)
    assert m["T_2m_C"] > m["T_surface_C"]   # ниже → теплее
    assert m["T_cloud_C"] is not None
    assert m["T_min_C"] < m["T_max_C"]


def test_extract_t_metrics_no_cloud():
    """Без облака T_cloud_C = None."""
    nz = 10
    z_mc = np.linspace(100.0, 15000.0, nz)
    theta = np.full(nz, 300.0)
    exner = np.linspace(1.0, 0.5, nz)
    qc = np.zeros(nz)

    m = extract_t_metrics(theta, exner, z_mc, qc)
    assert m["T_cloud_C"] is None