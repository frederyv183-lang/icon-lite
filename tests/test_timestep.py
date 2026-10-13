import sys
sys.path.insert(0, "src")
import numpy as np
from grid import make_vertical_grid
from state import make_reference_atmosphere
from timestep import step_forward

grid = make_vertical_grid(nlev=40, z_top=20000.0, stretch=1.15)
z_mc = grid["z_mc"]
z_ifc = grid["z_ifc"]
wgtfac_c = grid["wgtfac_c"]
inv_dz = 1.0 / grid["dz"]

ref = make_reference_atmosphere(z_mc, z_ifc, theta_surf=300.0)
rho = ref["rho_ref"].copy()
theta_ref = ref["theta_ref"]
exner_ref = ref["exner_ref"]

# тёплая и влажная аномалия одновременно
theta = theta_ref + 2.0 * np.exp(-((z_mc - 5000.0) / 1500.0) ** 2)
qv = 0.012 * np.exp(-((z_mc - 5000.0) / 1500.0) ** 2)  # 12 г/кг в центре

w = np.zeros(len(z_ifc))
pi_pr = np.zeros_like(rho)

print("=== Прогон 300 шагов dt=0.5 (150 с) с влажностью ===")
nsteps = 300
dt = 0.5
for n in range(nsteps):
    out = step_forward(rho, w, theta, qv, exner_ref, theta_ref,
                       wgtfac_c, inv_dz, dt)
    rho = out["rho"]; w = out["w"]; qv = out["qv"]; pi_pr = out["pi_pr"]
    if n % 100 == 0:
        print(f"  шаг {n:4d}: max|w| = {np.max(np.abs(w)):.4e}, "
              f"max qv = {np.max(qv)*1000:.2f} г/кг")

print()
print("=== Финальное состояние ===")
print("max|w| =", float(np.max(np.abs(w))), "м/с")
print("max qv =", float(np.max(qv))*1000, "г/кг")
idx = int(np.argmin(np.abs(z_ifc - 5000.0)))
print("w вокруг 5 км:", np.round(w[idx-1:idx+3], 3))
print("Знак положительный:", bool(w[idx] > 0))