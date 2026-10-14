"""нит-тесты для src/fog.py."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fog import (radiative_cooling_step, surface_cooling_step_2d,
                 diagnose_fog_flag, visibility_from_qc, fog_metrics)


def test_radiative_cooling_reduces_theta():
    """адиационное охлаждение уменьшает theta[0] при theta[0] > theta_ref."""
    theta = np.array([300.0, 295.0, 290.0])
    theta_ref = np.array([299.0, 295.0, 290.0])
    theta_new = radiative_cooling_step(theta, dt=1.0, k_rad=1e-3, theta_ref=theta_ref)
    assert theta_new[0] < theta[0]
    # ерхние уровни не меняются
    assert theta_new[1] == theta[1]
    assert theta_new[2] == theta[2]


def test_radiative_cooling_zero_when_equal():
    """ри theta[0] == theta_ref[0] охлаждения нет."""
    theta = np.array([300.0, 295.0])
    theta_ref = np.array([300.0, 295.0])
    theta_new = radiative_cooling_step(theta, dt=1.0, k_rad=1e-3, theta_ref=theta_ref)
    assert np.allclose(theta_new, theta)


def test_surface_cooling_2d_proportional_to_U():
    """ем больше U, тем сильнее охлаждение за шаг."""
    theta = np.array([[300.0, 300.0], [295.0, 295.0]])
    theta_surf = 295.0

    cool_slow = surface_cooling_step_2d(theta, dt=1.0, U=1.0, k_surf=1e-3,
                                        theta_surf=theta_surf)
    cool_fast = surface_cooling_step_2d(theta, dt=1.0, U=10.0, k_surf=1e-3,
                                        theta_surf=theta_surf)

    delta_slow = theta[0, 0] - cool_slow[0, 0]
    delta_fast = theta[0, 0] - cool_fast[0, 0]
    assert delta_fast > delta_slow
    # ерхний уровень не меняется
    assert np.allclose(cool_slow[1], theta[1])


def test_surface_cooling_zero_U():
    """ри U = 0 охлаждения нет."""
    theta = np.array([[300.0, 300.0], [295.0, 295.0]])
    theta_new = surface_cooling_step_2d(theta, dt=1.0, U=0.0, k_surf=1e-3,
                                        theta_surf=290.0)
    assert np.allclose(theta_new, theta)


def test_diagnose_fog_flag_true():
    """qc > 0 в приземном слое → туман."""
    z_mc = np.array([100.0, 300.0, 500.0])
    qc = np.array([2e-4, 0.0, 0.0])
    assert diagnose_fog_flag(qc, z_mc) is True


def test_diagnose_fog_flag_false():
    """qc = 0 везде → тумана нет."""
    z_mc = np.array([100.0, 300.0, 500.0])
    qc = np.array([0.0, 0.0, 0.0])
    assert diagnose_fog_flag(qc, z_mc) is False


def test_diagnose_fog_flag_only_high():
    """qc > 0 только выше 200 м → не туман."""
    z_mc = np.array([100.0, 300.0, 500.0])
    qc = np.array([0.0, 5e-4, 0.0])
    assert diagnose_fog_flag(qc, z_mc) is False


def test_visibility_formula():
    """vis ≈ 0.02 / qc, ограничена 20_000 м."""
    qc = np.array([1e-4, 1e-5, 1e-8])
    vis = visibility_from_qc(qc)
    assert abs(vis[0] - 200.0) < 1.0
    assert abs(vis[1] - 2000.0) < 10.0
    assert vis[2] == 20000.0


def test_fog_metrics_with_fog():
    """етрики тумана при qc > 0 в приземном слое."""
    z_mc = np.array([100.0, 300.0, 500.0])
    qc = np.array([1e-4, 0.0, 0.0])
    m = fog_metrics(qc, z_mc)
    assert m["has_fog"] is True
    assert m["vis_min_m"] < 1000.0
    assert m["qc_surface_gkg"] > 0.0
    assert m["top_fog_m"] > 0.0


def test_fog_metrics_no_fog():
    """етрики тумана при qc = 0."""
    z_mc = np.array([100.0, 300.0, 500.0])
    qc = np.zeros(3)
    m = fog_metrics(qc, z_mc)
    assert m["has_fog"] is False
    assert m["vis_min_m"] == 20000.0
    assert m["qc_surface_gkg"] == 0.0
    assert m["top_fog_m"] == 0.0