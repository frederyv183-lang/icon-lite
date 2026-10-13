import sys
sys.path.insert(0, "src")
import numpy as np
from grid2d import make_grid2d
from state2d import make_initial_2d

g = make_grid2d(nz=40, nx=80, x_total=40000.0, z_top=20000.0)
init = make_initial_2d(g, theta_ref=None, qv_max=0.020, dtheta=2.0,
                       x0_km=20.0, z0_km=3.0, radius_x_km=5.0, radius_z_km=1.5)

print("=== Размеры полей ===")
for key in ["rho", "w", "vn", "theta", "qv", "qc"]:
    arr = init[key]
    print(f"  {key:6s}: {arr.shape}")

print()
print("=== Значения в центре пузыря ===")
i0 = int(np.argmin(np.abs(g["x_mc"] - 20000.0)))
k0 = int(np.argmin(np.abs(g["z_mc"] - 3000.0)))
print(f"Индексы центра: i={i0}, k={k0}")
print(f"x = {g['x_mc'][i0]:.0f} м, z = {g['z_mc'][k0]:.0f} м")
print(f"theta там: {init['theta'][k0, i0]:.4f} К")
print(f"theta_ref там: {init['theta_ref_2d'][k0, i0]:.4f} К")
print(f"dtheta: {init['theta'][k0, i0] - init['theta_ref_2d'][k0, i0]:.4f} К (ожидаем 2.0)")
print(f"qv там: {init['qv'][k0, i0]*1000:.2f} г/кг (ожидаем 20.00)")
print()

print("=== Значения далеко от пузыря ===")
i_far = 0
k_far = 0
print(f"theta[i=0, k=0]: {init['theta'][k_far, i_far]:.4f} К")
print(f"qv[i=0, k=0]: {init['qv'][k_far, i_far]*1000:.6f} г/кг (ожидаем ~0)")
print()

print("=== 2D-структура: max по каждому уровню ===")
for k in [0, 5, 10, 15, 20, 30, 39]:
    z = g["z_mc"][k]
    max_theta = float(np.max(init['theta'][k, :]))
    dth = max_theta - 300.0
    print(f"  k={k:2d}, z={z:6.0f} м: max dθ = {dth:6.3f} К")