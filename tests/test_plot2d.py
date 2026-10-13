import sys
sys.path.insert(0, "src")
import numpy as np
from grid2d import make_grid2d
from state2d import make_initial_2d
from timestep2d import step_forward_2d
from plotting2d import make_snapshots_2d
from state import diagnose_exner
from thermo import virtual_theta
import base64

g = make_grid2d(nz=40, nx=80, x_total=40000.0, z_top=20000.0)
init = make_initial_2d(g, None, qv_max=0.020, dtheta=2.0,
                       x0_km=20.0, z0_km=3.0, radius_x_km=5.0, radius_z_km=1.5)

theta_v_ref = virtual_theta(init["theta_ref_2d"], np.zeros_like(init["qv"]))
exner_ref_2d = diagnose_exner(init["rho"], theta_v_ref)

rho = init["rho"].copy()
w = init["w"].copy()
vn = init["vn"].copy()
theta = init["theta"].copy()
qv = init["qv"].copy()
qc = init["qc"].copy()

nsteps = 800
dt = 0.5
n_snap = 8
snap_every = nsteps // n_snap

times = [0.0]
w_hist = [w.copy()]
qc_hist = [qc.copy()]

for n in range(nsteps):
    out = step_forward_2d(rho, w, vn, theta, qv, qc,
                          exner_ref_2d, init["theta_ref_2d"], g, dt)
    rho = out["rho"]; w = out["w"]; vn = out["vn"]
    theta = out["theta"]; qv = out["qv"]; qc = out["qc"]
    if (n + 1) % snap_every == 0 and len(times) < n_snap:
        times.append((n + 1) * dt)
        w_hist.append(w.copy())
        qc_hist.append(qc.copy())

print(f"Прогон завершён. Финальное max|w| = {np.max(np.abs(w)):.3f}")
print(f"max|vn| = {np.max(np.abs(vn)):.3f}")
print(f"max qc = {np.max(qc)*1000:.4f} г/кг")

b64 = make_snapshots_2d(w_hist, qc_hist,
                        g["x_mc"], g["z_ifc"], g["z_mc"], times)
with open("snapshots2d.png", "wb") as f:
    f.write(base64.b64decode(b64))
print("Сохранено: snapshots2d.png")