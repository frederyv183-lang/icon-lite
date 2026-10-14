"""Tests for src/state3d.py."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grid3d import make_grid3d
from state3d import make_initial_3d


def test_state3d_shapes():
    g = make_grid3d(nx=10, ny=8, nz=6, x_total=20000.0, y_total=16000.0)
    init = make_initial_3d(g, dtheta=2.0, qv_max=0.015,
                           z0_km=3.0, radius_x_km=3.0,
                           radius_y_km=3.0, radius_z_km=1.5)
    for key in ("theta", "qv", "qc", "rho", "u", "v", "theta_ref_3d"):
        assert init[key].shape == (6, 8, 10), key
    assert init["w"].shape == (7, 8, 10)
    # quiescent initial velocity
    assert np.all(init["u"] == 0.0)
    assert np.all(init["v"] == 0.0)
    assert np.all(init["w"] == 0.0)
    assert np.all(init["qc"] == 0.0)


def test_state3d_positive_anomaly_at_center():
    g = make_grid3d(nx=30, ny=30, nz=20,
                    x_total=20000.0, y_total=20000.0)
    init = make_initial_3d(g, dtheta=2.0, qv_max=0.015,
                           z0_km=3.0, radius_x_km=3.0,
                           radius_y_km=3.0, radius_z_km=1.5)
    anomaly = init["theta"] - init["theta_ref_3d"]
    # peak close to dtheta (discretisation slack)
    assert float(np.max(anomaly)) > 1.5
    assert float(np.max(anomaly)) <= 2.0 + 1e-9
    # nowhere negative
    assert float(np.min(anomaly)) >= 0.0
    # away from bubble - almost zero
    assert float(anomaly[0, 0, 0]) < 1e-3


def test_state3d_custom_theta_ref():
    g = make_grid3d(nx=8, ny=8, nz=6, x_total=20000.0, y_total=20000.0)
    theta_ref = np.linspace(295.0, 320.0, 6)
    init = make_initial_3d(g, theta_ref=theta_ref, dtheta=1.0,
                           qv_max=0.0, z0_km=2.0,
                           radius_x_km=2.0, radius_y_km=2.0,
                           radius_z_km=1.0)
    # background matches theta_ref everywhere except bubble
    far = init["theta"][0, 0, 0]
    assert float(far) == pytest.approx(float(theta_ref[0]), abs=1e-6)
