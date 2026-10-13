import sys
sys.path.insert(0, "src")
import numpy as np
from grid import make_vertical_grid
from state import make_reference_atmosphere
from tracer import dqv_dt, step_qv

grid = make_vertical_grid(nlev=20, z_top=20000.0, stretch=1.15)
z_mc = grid["z_mc"]
z_ifc = grid["z_ifc"]
inv_dz = 1.0 / grid["dz"]

ref = make_reference_atmosphere(z_mc, z_ifc, theta_surf=300.0)
rho = ref["rho_ref"]

# --- ТЕСТ 1: однородная влажность + покой w=0 -> dqv/dt = 0 ---
qv_uniform = np.full_like(z_mc, 0.010)  # 10 г/кг везде
w_zero = np.zeros(len(z_ifc))
dqv = dqv_dt(rho, qv_uniform, w_zero, inv_dz)
print("=== ТЕСТ 1: однородная qv, w=0 ===")
print("dqv/dt (макс модуль):", float(np.max(np.abs(dqv))))
print("Ожидание: 0")

# --- ТЕСТ 2: qv = 0 везде -> dqv/dt = 0 ---
qv_zero = np.zeros_like(z_mc)
dqv_zero = dqv_dt(rho, qv_zero, w_zero, inv_dz)
print()
print("=== ТЕСТ 2: qv=0 везде ===")
print("dqv/dt (макс модуль):", float(np.max(np.abs(dqv_zero))))
print("Ожидание: 0")

# --- ТЕСТ 3: гауссово облако qv в середине, w=0 ---
# без потока ничего не двигается -> dqv/dt = 0
qv_gauss = 0.015 * np.exp(-((z_mc - 5000.0) / 1500.0) ** 2)
dqv_gauss = dqv_dt(rho, qv_gauss, w_zero, inv_dz)
print()
print("=== ТЕСТ 3: гауссово облако qv, w=0 ===")
print("dqv/dt (макс модуль):", float(np.max(np.abs(dqv_gauss))))
print("Ожидание: 0")

# --- ТЕСТ 4: однородный подъём w=+1 м/с -> облако сдвигается вверх ---
w_up = np.ones(len(z_ifc))
qv_new = step_qv(rho, qv_gauss, w_up, inv_dz, dt=60.0)
print()
print("=== ТЕСТ 4: подъём w=+1 м/с за 60 с ===")
idx_old = int(np.argmax(qv_gauss))
idx_new = int(np.argmax(qv_new))
print(f"Максимум qv до:    z = {z_mc[idx_old]:.0f} м, qv = {qv_gauss[idx_old]:.4f}")
print(f"Максимум qv после: z = {z_mc[idx_new]:.0f} м, qv = {qv_new[idx_new]:.4f}")
print("Ожидание: облако сдвинулось вверх на ~60 м")
print(f"Сдвиг: {z_mc[idx_new] - z_mc[idx_old]:.0f} м")

# --- ТЕСТ 5: сохранение общей массы воды ---
mass_old = float(np.sum(rho * qv_gauss * grid["dz"]))
mass_new = float(np.sum(rho * qv_new * grid["dz"]))
print()
print("=== ТЕСТ 5: сохранение массы воды ===")
print(f"Масса воды до:    {mass_old:.6e} кг/м²")
print(f"Масса воды после: {mass_new:.6e} кг/м²")
print(f"Относительное изменение: {100*(mass_new/mass_old - 1):.4f}%")
print("Ожидание: < 0.1% (немного из-за численной диффузии)")