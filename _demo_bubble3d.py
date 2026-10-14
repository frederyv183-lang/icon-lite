"""Render the 3D bubble demo to bubble_3d_demo.png."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from grid3d import make_grid3d
from state3d import make_initial_3d
from plotting3d import make_bubble_3d_bytes

g = make_grid3d(nx=30, ny=30, nz=20,
                x_total=20000.0, y_total=20000.0)

init = make_initial_3d(g, dtheta=2.0, qv_max=0.015,
                       z0_km=3.0,
                       radius_x_km=3.0, radius_y_km=3.0,
                       radius_z_km=1.5)

png = make_bubble_3d_bytes(init["theta"], g, dark=False,
                           theta_ref=init["theta_ref_3d"],
                           title="3D warm bubble (dtheta=2 K)")

out = ROOT / "bubble_3d_demo.png"
out.write_bytes(png)
print("wrote:", out)
print("size:", len(png), "bytes")
