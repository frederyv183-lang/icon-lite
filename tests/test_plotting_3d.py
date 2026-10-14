"""Smoke tests for 3D visualization."""
import base64
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _make_cape(nz=20, nx=30):
    x_mc = np.linspace(0.0, 20000.0, nx)
    z_mc = np.linspace(0.0, 20000.0, nz)
    rng = np.random.default_rng(0)
    cape = rng.random((nz, nx)) * 1000.0
    return x_mc, z_mc, cape


def test_make_storm2d_map_3d_returns_png():
    from plotting_storm2d import make_storm2d_map_3d
    x_mc, z_mc, cape = _make_cape()
    png = make_storm2d_map_3d(x_mc, z_mc, cape, dark=False)
    assert isinstance(png, bytes)
    assert png[:8] == PNG_SIG
    assert len(png) > 5000


def test_make_snapshots_2d_3d_returns_base64_png():
    from plotting2d import make_snapshots_2d_3d
    nz, nx = 20, 30
    x_mc = np.linspace(0.0, 20000.0, nx)
    z_ifc = np.linspace(0.0, 20000.0, nz + 1)
    z_mc = 0.5 * (z_ifc[:-1] + z_ifc[1:])
    times = [0.0, 100.0, 200.0, 300.0]
    rng = np.random.default_rng(1)
    w_hist = [rng.standard_normal((nz + 1, nx)) * 2.0 for _ in times]
    qc_hist = [np.abs(rng.standard_normal((nz, nx))) * 1e-3 for _ in times]
    qr_hist = [np.abs(rng.standard_normal((nz, nx))) * 1e-4 for _ in times]

    b64 = make_snapshots_2d_3d(
        w_hist, qc_hist, qr_hist,
        x_mc, z_ifc, z_mc, times, dark=False,
    )
    assert isinstance(b64, str)
    raw = base64.b64decode(b64)
    assert raw[:8] == PNG_SIG
    assert len(raw) > 5000


def test_download_routes_return_png():
    import app as app_mod
    client = app_mod.app.test_client()

    r1 = client.get(
        "/download/storm2d.map3d.png"
        "?nz=20&nx=30&x_total=15&dtheta=2&qv_max=15"
        "&zbubble=2&radius_x=4&dt=0.5&nsteps=100&use_diffusion=1"
    )
    assert r1.status_code == 200
    assert r1.headers.get("Content-Type", "").startswith("image/png")
    assert r1.data[:8] == PNG_SIG

    r2 = client.get(
        "/download/2d.3d.png"
        "?nz=20&nx=40&x_total=20&dtheta=2&qv_max=15"
        "&zbubble=2&radius_x=4&dt=0.5&nsteps=200"
    )
    assert r2.status_code == 200
    assert r2.headers.get("Content-Type", "").startswith("image/png")
    assert r2.data[:8] == PNG_SIG