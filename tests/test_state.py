import sys
sys.path.insert(0, "src")
import numpy as np
from grid import make_vertical_grid
from state import make_reference_atmosphere, diagnose_exner

# сетка
grid = make_vertical_grid(nlev=20, z_top=20000.0, stretch=1.15)
z_mc = grid["z_mc"]
z_ifc = grid["z_ifc"]

# опорная атмосфера
ref = make_reference_atmosphere(z_mc, z_ifc, theta_surf=300.0, p_surf=101325.0)

print("=== Опорная атмосфера ===")
print("Высоты (м), первые 3:", np.round(z_mc[:3], 1))
print("Высоты (м), последние 3:", np.round(z_mc[-3:], 1))
print()
print("exner_ref, первые 3:", np.round(ref["exner_ref"][:3], 5))
print("exner_ref, последние 3:", np.round(ref["exner_ref"][-3:], 5))
print()
print("rho_ref, первые 3 (кг/м³):", np.round(ref["rho_ref"][:3], 4))
print("rho_ref, последние 3 (кг/м³):", np.round(ref["rho_ref"][-3:], 5))
print()
print("theta_ref, первые 3 (К):", np.round(ref["theta_ref"][:3], 1))
print()

# проверка: диагностика Exner из rho, theta
rho_test = ref["rho_ref"].reshape(-1, 1)      # одна колонка
theta_test = ref["theta_ref"].reshape(-1, 1)
exner_diag = diagnose_exner(rho_test, theta_test)

print("=== Проверка diagnose_exner ===")
print("exner_diag, первые 3:", np.round(exner_diag[:3, 0], 5))
print("exner_ref, первые 3: ", np.round(ref["exner_ref"][:3], 5))
print("Максимальная разница:", float(np.max(np.abs(exner_diag[:, 0] - ref["exner_ref"]))))
