"""Smoke tests for src/app_3d_helper.py."""
import base64
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PNG_SIG = b"\x89PNG\r\n\x1a\n"


def test_run_model_3d_static_returns_png():
    from app_3d_helper import run_model_3d
    out = run_model_3d(nx=12, ny=12, nz=8,
                       x_total=20, y_total=20, z_top=15,
                       dtheta=2.0, qv_max=0.0, z0=3.0,
                       radius_x=3.0, radius_y=3.0, radius_z=1.5)
    assert "image_initial" in out
    raw = base64.b64decode(out["image_initial"])
    assert raw[:8] == PNG_SIG
    assert len(raw) > 5000
    assert out["max_dtheta"] > 1.5
    assert out["min_dtheta"] >= 0.0
    assert out["n_cells"] == 12 * 12 * 8


def test_run_model_3d_rising_returns_png():
    from app_3d_helper import run_model_3d_rising
    out = run_model_3d_rising(nx=12, ny=12, nz=8,
                              x_total=20, y_total=20, z_top=15,
                              dtheta=2.0, qv_max=0.0, z0=2.0,
                              radius_x=3.0, radius_y=3.0, radius_z=1.5,
                              dt=0.5, nsteps=20,
                              use_pressure=False,
                              n_snapshots=6)
    assert "image_rising" in out
    raw = base64.b64decode(out["image_rising"])
    assert raw[:8] == PNG_SIG
    assert len(raw) > 5000
    assert out["final_t"] == 10.0
    assert out["n_snapshots"] >= 2


def test_flask_3d_route_get():
    import sys as _sys
    if str(ROOT) not in _sys.path:
        _sys.path.insert(0, str(ROOT))
    import app as app_mod
    c = app_mod.app.test_client()
    r = c.get("/3d")
    assert r.status_code == 200
    assert b"3D" in r.data
