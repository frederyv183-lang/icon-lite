import sys
sys.path.insert(0, "src")
import numpy as np
from grid import make_vertical_grid
from state import make_reference_atmosphere
from exner import perturbation, exner_at_ifc, vertical_gradient

# сетка
grid = make_vertical_grid(nlev=20, z_top=20000.0, stretch=1.15)
z_mc = grid["z_mc"]
z_ifc = grid["z_ifc"]
wgtfac_c = grid["wgtfac_c"]

# опорное состояние
ref = make_reference_atmosphere(z_mc, z_ifc, theta_surf=300.0)
exner_ref = ref["exner_ref"]

# сделаем тестовое возмущение Exner: +1% на поверхности, 0 на верхней границе
exner_test = exner_ref * (1.0 + 0.01 * (1.0 - z_mc / z_mc[-1]))

# возмущение
pi_pr = perturbation(exner_test, exner_ref)
print("=== Возмущение Exner ===")
print("pi_pr, первые 3:", np.round(pi_pr[:3], 6))
print("pi_pr, последние 3:", np.round(pi_pr[-3:], 6))
print()

# Exner на интерфейсах
pi_ifc = exner_at_ifc(pi_pr, wgtfac_c)
print("=== Exner на интерфейсах ===")
print("Форма массива pi_ifc:", pi_ifc.shape)  # ожидаем (21,)
print("pi_ifc, первые 5:", np.round(pi_ifc[:5], 6))
print("pi_ifc, последние 5:", np.round(pi_ifc[-5:], 6))
print()

# вертикальный градиент
inv_dz = 1.0 / grid["dz"]
dpi_dz = vertical_gradient(pi_ifc, inv_dz)
print("=== Вертикальный градиент dpi/dz ===")
print("Форма массива dpi_dz:", dpi_dz.shape)  # ожидаем (20,)
print("dpi_dz, первые 3:", np.round(dpi_dz[:3], 8))
print("dpi_dz, последние 3:", np.round(dpi_dz[-3:], 8))
print()
print("Знак dpi_dz (ожидаем положительный, т.к. pi' убывает с высотой):")
print("  все положительные:", bool(np.all(dpi_dz > 0)))
