"""Tests for src/timestep3d.py (collocation)."""
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
    g = make_grid3d(nx=nx, ny=ny, nz=nz, x_total=20000.0, y_total=20000.0)
    init = make_initial_3d(g, dtheta=dtheta, qv_max=0.0,
                           z0_km=3.0, radius_x_km=3.0,
                           radius_y_km=3.0, radius_z_km=1.5)
    return g, init


def test_step3d_shapes():
    g, s = _mk_state()
    out = step_forward_3d(s, g, dt=1.0)
    for key in ("theta", "qv", "u", "v", "w", "rho", "theta_ref_3d"):
        assert out[key].shape == (g["nz"], g["ny"], g["nx"]), key


def test_step3d_quiescent():
    g, s = _mk_state(dtheta=0.0)
    out = step_forward_3d(s, g, dt=1.0)
    assert np.allclose(out["w"], 0.0, atol=1e-12)
    assert np.allclose(out["theta"], s["theta"], atol=1e-12)


def test_step3d_bubble_rises():
    g, s = _mk_state(dtheta=2.0)
    cur = s
    for _ in range(20):
        cur = step_forward_3d(cur, g, dt=0.5)
    assert float(np.max(cur["w"])) > 0.0


def test_step3d_cfl_warning():
    g, s = _mk_state()
    s2 = dict(s)
    s2["u"] = np.full_like(s["u"], 3000.0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        step_forward_3d(s2, g, dt=1.0)
    assert any("CFL" in str(x.message) for x in w)


def test_step3d_use_pressure():
    from grid3d import make_grid3d
    from state3d import make_initial_3d
    g = make_grid3d(nx=8, ny=8, nz=8, x_total=20000.0, y_total=20000.0,
                    z_top=8000.0, stretch=1.0)
    s = make_initial_3d(g, dtheta=2.0, qv_max=0.0,
                        z0_km=2.0, radius_x_km=3.0,
                        radius_y_km=3.0, radius_z_km=1.5)
    out_plain = step_forward_3d(s, g, dt=0.5)
    out_press = step_forward_3d(s, g, dt=0.5,
                                use_pressure=True, n_iter_pressure=500)
    for key in ('theta', 'qv', 'u', 'v', 'w'):
        assert out_plain[key].shape == out_press[key].shape
    assert np.all(np.isfinite(out_press['u']))
    assert np.all(np.isfinite(out_press['v']))
    assert np.all(np.isfinite(out_press['w']))
