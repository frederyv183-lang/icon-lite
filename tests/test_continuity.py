import sys
sys.path.insert(0, "src")
import numpy as np
from grid import make_vertical_grid
from state import make_reference_atmosphere
from continuity import drho_dt, step_euler

grid = make_vertical_grid(nlev=20, z_top=20000.0, stretch=1.15)
z_mc = grid["z_mc"]
z_ifc = grid["z_ifc"]
inv_dz = 1.0 / grid["dz"]

ref = make_reference_atmosphere(z_mc, z_ifc, theta_surf=300.0)
rho_ref = ref["rho_ref"]

# --- ТЕСТ 1: rho покоится, w = 0 -> drho/dt = 0 ---
w_zero = np.zeros(len(z_ifc))
drho_zero = drho_dt(rho_ref, w_zero, inv_dz)
print("=== ТЕСТ 1: покой ===")
print("drho/dt (должно быть 0):", np.round(drho_zero[:3], 12))
print("Максимум |drho/dt|:", float(np.max(np.abs(drho_zero))))
print()

# --- ТЕСТ 2: однородный восходящий поток w = +1 м/с везде ---
w_up = np.ones(len(z_ifc)) * 1.0
drho_up = drho_dt(rho_ref, w_up, inv_dz)
print("=== ТЕСТ 2: однородный подъём w=+1 м/с ===")
print("drho/dt (кг/(м3·с)), первые 3:", np.round(drho_up[:3], 8))
print("drho/dt, последние 3:", np.round(drho_up[-3:], 8))
print()
print("Ожидание: drho/dt отрицательное внизу (воздух уходит вверх),")
print("          положительное вверху (воздух приходит снизу).")
print("Знак внизу:", "отриц." if drho_up[0] < 0 else "полож.",
      "| Знак вверху:", "полож." if drho_up[-1] > 0 else "отриц.")
print()

# --- ТЕСТ 3: сходимость (конвергенция) w сверху вниз ---
# w уменьшается с высотой -> воздух «сжимается» -> rho растёт
w_conv = np.linspace(2.0, 0.0, len(z_ifc))  # от 2 м/с внизу до 0 вверху
drho_conv = drho_dt(rho_ref, w_conv, inv_dz)
print("=== ТЕСТ 3: конвергенция (w падает с высотой) ===")
print("drho/dt, первые 3:", np.round(drho_conv[:3], 8))
print("drho/dt, последние 3:", np.round(drho_conv[-3:], 8))
print("Ожидание: drho/dt > 0 (воздух накапливается).")
print("Все положительные:", bool(np.all(drho_conv > 0)))
print()

# --- ТЕСТ 4: шаг по времени 10 с ---
rho_new = step_euler(rho_ref, drho_conv, 10.0)
print("=== ТЕСТ 4: шаг 10 с ===")
print("rho было:", np.round(rho_ref[:3], 6))
print("rho стало:", np.round(rho_new[:3], 6))
print("Разница (%):", np.round(100 * (rho_new[:3] - rho_ref[:3]) / rho_ref[:3], 4))
