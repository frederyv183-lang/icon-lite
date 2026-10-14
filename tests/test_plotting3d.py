"""Smoke test for src/plotting3d.py."""
import base64
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PNG_SIG = b"\x89PNG\r\n\x1a\n"


def test_make_bubble_3d_returns_png():
    from grid3d import make_grid3d
    from state3d import make_initial_3d
    from plotting3d import make_bubble_3d, make_bubble_3d_bytes

    g = make_grid3d(nx=12, ny=12, nz=8,
                    x_total=20000.0, y_total=20000.0)
    init = make_initial_3d(g, dtheta=2.0, qv_max=0.015,
                           z0_km=3.0, radius_x_km=3.0,
                           radius_y_km=3.0, radius_z_km=1.5)

    b64 = make_bubble_3d(init["theta"], g, dark=False,
                         theta_ref=init["theta_ref_3d"])
    assert isinstance(b64, str)
    raw = base64.b64decode(b64)
    assert raw[:8] == PNG_SIG
    assert len(raw) > 5000

    png = make_bubble_3d_bytes(init["theta"], g, dark=False,
                               theta_ref=init["theta_ref_3d"])
    assert isinstance(png, bytes)
    assert png[:8] == PNG_SIG
