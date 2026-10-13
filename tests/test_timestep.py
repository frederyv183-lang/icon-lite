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

# сильное влажное возмущение: +2 K и +15 г/кг в центре
shape = np.exp(-((z_mc - 5000.0) / 1500.0) ** 2)
theta = theta_ref + 2.0 * shape
qv = 0.015 * shape
qc = np.zeros_like(theta)

w = np.zeros(len(z_ifc))

print("=== Прогон 300 шагов dt=0.5 с конденсацией ===")
nsteps = 300
dt = 0.5
for n in range(nsteps):
    out = step_forward(rho, w, theta, qv, qc, exner_ref, theta_ref,
                       wgtfac_c, inv_dz, dt)
    rho = out["rho"]; w = out["w"]
    theta = out["theta"]; qv = out["qv"]; qc = out["qc"]
    if n % 50 == 0:
        print(f"  шаг {n:4d}: max|w| = {np.max(np.abs(w)):7.3f} м/с | "
              f"max qv = {np.max(qv)*1000:5.2f} г/кг | "
              f"max qc = {np.max(qc)*1000:5.2f} г/кг | "
              f"max dθ = {np.max(theta-theta_ref):5.2f} K")

print()
print("=== Финальное состояние ===")
print("max|w| =", float(np.max(np.abs(w))), "м/с")
print("max qv =", float(np.max(qv))*1000, "г/кг")
print("max qc =", float(np.max(qc))*1000, "г/кг")
print("max dtheta =", float(np.max(theta - theta_ref)), "K")
print()
idx = int(np.argmin(np.abs(z_ifc - 5000.0)))
print("w вокруг 5 км:", np.round(w[idx-1:idx+3], 3))
print("qc вокруг 5 км (г/кг):", np.round(qc[idx-1:idx+3]*1000, 3))
i_max = int(np.argmax(w))
print(f"Максимум w на z = {z_ifc[i_max]:.0f} м")