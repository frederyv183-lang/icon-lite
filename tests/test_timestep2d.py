import sys
sys.path.insert(0, "src")
import numpy as np
from grid2d import make_grid2d
from state2d import make_initial_2d
from timestep2d import step_forward_2d

g = make_grid2d(nz=40, nx=80, x_total=40000.0, z_top=20000.0)
init = make_initial_2d(g, None, qv_max=0.020, dtheta=2.0,
                       x0_km=20.0, z0_km=3.0, radius_x_km=5.0, radius_z_km=1.5)

# опорный Exner 2D
from state import diagnose_exner
from thermo import virtual_theta
theta_v_ref = virtual_theta(init["theta_ref_2d"], np.zeros_like(init["qv"]))
exner_ref_2d = diagnose_exner(init["rho"], theta_v_ref)

rho = init["rho"].copy()
w = init["w"].copy()
vn = init["vn"].copy()
theta = init["theta"].copy()
qv = init["qv"].copy()
qc = init["qc"].copy()

print("=== Прогон 200 шагов 2D, dt=0.5 с (100 с) ===")
nsteps = 200
dt = 0.5
for n in range(nsteps):
    out = step_forward_2d(rho, w, vn, theta, qv, qc,
                          exner_ref_2d, init["theta_ref_2d"],
                          g, dt)
    rho = out["rho"]; w = out["w"]; vn = out["vn"]
    theta = out["theta"]; qv = out["qv"]; qc = out["qc"]
    if n % 50 == 0:
        print(f"  шаг {n:4d}: max|w| = {np.max(np.abs(w)):.4f} | "
              f"max|vn| = {np.max(np.abs(vn)):.4f} | "
              f"max qc = {np.max(qc)*1000:.3f} | "
              f"max dθ = {np.max(theta - init['theta_ref_2d']):.3f}")

print()
print("=== Финальное состояние ===")
print(f"max|w|  = {np.max(np.abs(w)):.4f} м/с")
print(f"max|vn| = {np.max(np.abs(vn)):.4f} м/с")
print(f"max qc  = {np.max(qc)*1000:.4f} г/кг")
print(f"max dθ  = {np.max(theta - init['theta_ref_2d']):.4f} K")
print()
print("Проверка на NaN/Inf:", np.any(~np.isfinite(w)), np.any(~np.isfinite(vn)))