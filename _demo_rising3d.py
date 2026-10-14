"""Render 6-panel animation of a rising 3D bubble.

Each panel shows the x-z slice through the bubble centre at that time.
Saved to bubble_3d_rising.png.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from grid3d import make_grid3d
from state3d import make_initial_3d
from timestep3d import step_forward_3d

g = make_grid3d(nx=24, ny=24, nz=16,
                x_total=20000.0, y_total=20000.0)

init = make_initial_3d(g, dtheta=2.0, qv_max=0.0,
                       z0_km=2.0,
                       radius_x_km=3.0, radius_y_km=3.0,
                       radius_z_km=1.5)

theta_ref = init["theta_ref_3d"]
state = init
snapshots = [(0.0, state["theta"].copy())]

dt = 0.5
n_steps = 200
n_snap = 6
every = max(1, n_steps // n_snap)

t = 0.0
snap_i = 0
for step in range(1, n_steps + 1):
    state = step_forward_3d(state, g, dt=dt)
    t += dt
    if step % every == 0 and snap_i < n_snap - 1:
        snapshots.append((t, state["theta"].copy()))
        snap_i += 1

# plot 6 panels of x-z slice at y = centre
j0 = g["ny"] // 2
x_km = g["x_mc"] / 1000.0
z_km = g["z_mc"] / 1000.0

fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.0))
fig.suptitle("3D warm bubble rising (x-z slice, y = centre)", fontsize=12)

vmax = float(np.max(np.abs(init["theta"] - theta_ref))) or 1.0
levels = np.linspace(0.0, vmax, 12)

for ax, (t_snap, theta_snap) in zip(axes.ravel(), snapshots):
    anom = theta_snap - theta_ref
    slice_xz = anom[:, j0, :]
    im = ax.pcolormesh(x_km, z_km, slice_xz,
                       cmap="YlOrRd", vmin=0.0, vmax=vmax,
                       shading="nearest")
    ax.set_xlabel("x, km")
    ax.set_ylabel("z, km")
    ax.set_title("t = %.0f s" % t_snap)
    ax.set_aspect("auto")

fig.subplots_adjust(left=0.05, right=0.92, top=0.92, bottom=0.08,
                    wspace=0.25, hspace=0.35)
cax = fig.add_axes([0.935, 0.15, 0.012, 0.70])
cb = fig.colorbar(im, cax=cax)
cb.set_label("theta - theta_ref, K")

out = ROOT / "bubble_3d_rising.png"
fig.savefig(out, dpi=110)
print("wrote:", out)
print("steps:", n_steps, "dt:", dt, "final t:", t)
print("max |w| at end:", float(np.max(np.abs(state["w"]))))
