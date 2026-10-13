import sys
sys.path.insert(0, "src")
import numpy as np
from grid import make_vertical_grid
from state import make_reference_atmosphere
from timestep import step_forward
from animation import make_snapshots_frame

grid = make_vertical_grid(nlev=40, z_top=20000.0, stretch=1.15)
z_mc = grid["z_mc"]
z_ifc = grid["z_ifc"]
wgtfac_c = grid["wgtfac_c"]
inv_dz = 1.0 / grid["dz"]

ref = make_reference_atmosphere(z_mc, z_ifc, theta_surf=300.0)
rho = ref["rho_ref"].copy()
theta_ref = ref["theta_ref"]
exner_ref = ref["exner_ref"]

theta_v = theta_ref + 2.0 * np.exp(-((z_mc - 5000.0) / 1500.0) ** 2)
w = np.zeros(len(z_ifc))
pi_pr = np.zeros_like(rho)

n_snap = 8
nsteps = 1200
dt = 0.5
snap_every = nsteps // n_snap

times = [0.0]
w_hist = [w.copy()]
rho_hist = [rho.copy()]
pi_hist = [pi_pr.copy()]

for n in range(nsteps):
    out = step_forward(rho, w, theta_v, exner_ref, theta_ref,
                       wgtfac_c, inv_dz, dt)
    rho = out["rho"]; w = out["w"]; pi_pr = out["pi_pr"]
    if (n + 1) % snap_every == 0 and len(times) < n_snap:
        times.append((n + 1) * dt)
        w_hist.append(w.copy())
        rho_hist.append(rho.copy())
        pi_hist.append(pi_pr.copy())

b64 = make_snapshots_frame(w_hist, rho_hist, pi_hist, z_ifc, z_mc, times)
print("PNG base64 длина:", len(b64))
with open("snapshots.png", "wb") as f:
    import base64
    f.write(base64.b64decode(b64))
print("Сохранено: snapshots.png")