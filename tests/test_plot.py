import sys
sys.path.insert(0, "src")
import numpy as np
from grid import make_vertical_grid
from state import make_reference_atmosphere
from timestep import step_forward
from plotting import plot_profiles

grid = make_vertical_grid(nlev=40, z_top=20000.0, stretch=1.15)
z_mc = grid["z_mc"]
z_ifc = grid["z_ifc"]
wgtfac_c = grid["wgtfac_c"]
inv_dz = 1.0 / grid["dz"]

ref = make_reference_atmosphere(z_mc, z_ifc, theta_surf=300.0)
rho = ref["rho_ref"].copy()
theta_ref = ref["theta_ref"]
exner_ref = ref["exner_ref"]

theta_v = theta_ref + 2.0 * np.exp(-((z_mc - 5000.0) / 1500.0)**2)
w = np.zeros(len(z_ifc))

# прогон с сохранением снимков
times = [0, 100, 300, 600]
w_hist, rho_hist, pi_hist = [], [], []
pi_pr = np.zeros_like(rho)
w_hist.append(w.copy()); rho_hist.append(rho.copy()); pi_hist.append(pi_pr.copy())

dt = 0.5
nsteps_total = 1200
snap_every = 200
snap_idx = 0
for n in range(nsteps_total):
    out = step_forward(rho, w, theta_v, exner_ref, theta_ref,
                       wgtfac_c, inv_dz, dt)
    rho = out["rho"]; w = out["w"]; pi_pr = out["pi_pr"]
    if (n + 1) % snap_every == 0:
        snap_idx += 1
        if snap_idx < len(times):
            w_hist.append(w.copy()); rho_hist.append(rho.copy()); pi_hist.append(pi_pr.copy())

out_path = plot_profiles(z_ifc, z_mc, w_hist, rho_hist, pi_hist, times,
                        savepath="output.png")
print("Сохранено:", out_path)
print("max|w| финал:", float(np.max(np.abs(w))))
