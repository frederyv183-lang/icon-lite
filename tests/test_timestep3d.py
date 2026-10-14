"""Tests for src/timestep3d.py."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grid3d import make_grid3d
from state3d import make_initial_3d
from timestep3d import step_forward_3d


def _mk_state(nx=8, ny=8, nz=6, dtheta=2.0):
    g = make_grid3d(nx=nx, ny=ny, nz=nz,
                    x_total=20000.0, y_total=20000.0)
    init = make_initial_3d(g, dtheta=dtheta, qv_max=0.0,
                           z0_km=3.0, radius_x_km=3.0,
                           radius_y_km=3.0, radius_z_km=1.5)
    return g, init


def test_step3d_shapes():
    g, s = _mk_state()
    out = step_forward_3d(s, g, dt=1.0)
    for key in ("theta", "qv", "u", "v", "rho", "theta_ref_3d"):
        assert out[key].shape == (g["nz"], g["ny"], g["nx"])
    assert out["w"].shape == (g["nz"] + 1, g["ny"], g["nx"])


def test_step3d_quiescent_stays_quiescent():
    g, s = _mk_state(dtheta=0.0)   # no anomaly -> no buoyancy
    out = step_forward_3d(s, g, dt=1.0)
    assert np.allclose(out["w"], 0.0, atol=1e-12)
    assert np.allclose(out["theta"], s["theta"], atol=1e-12)


def test_step3d_warm_bubble_rises():
    g, s = _mk_state(dtheta=2.0)
    # integrate 20 steps with small dt
    cur = s
    for _ in range(20):
        cur = step_forward_3d(cur, g, dt=0.5)
    max_w = float(np.max(cur["w"]))
    assert max_w > 0.0, "warm bubble must produce positive vertical velocity"
    # in the first steps the peak anomaly should still be near the center
    peak_idx = np.unravel_index(np.argmax(cur["w"]), cur["w"].shape)
    assert 0 < peak_idx[0] < g["nz"], "peak w should be inside the domain"


def test_step3d_cfl_warning():
    g, s = _mk_state()
    s2 = dict(s)
    s2["u"] = np.full_like(s["u"], 3000.0)  # huge velocity -> CFL break
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        step_forward_3d(s2, g, dt=1.0)
    assert any("CFL" in str(x.message) for x in w), "CFL warning expected"
