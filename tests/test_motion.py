import sys
sys.path.insert(0, "src")
import numpy as np
from grid import make_vertical_grid
from state import make_reference_atmosphere
from exner import perturbation, exner_at_ifc, vertical_gradient
from motion import dw_dt_vertical, step_euler

grid = make_vertical_grid(nlev=20, z_top=20000.0, stretch=1.15)
z_mc = grid["z_mc"]
z_ifc = grid["z_ifc"]
wgtfac_c = grid["wgtfac_c"]
inv_dz = 1.0 / grid["dz"]

ref = make_reference_atmosphere(z_mc, z_ifc, theta_surf=300.0)
exner_ref = ref["exner_ref"]
theta_ref = ref["theta_ref"]

# ТЕСТ 1: опорное состояние
pi_pr_zero = np.zeros_like(z_mc)
pi_ifc_zero = exner_at_ifc(pi_pr_zero, wgtfac_c)
dpi_dz_zero = vertical_gradient(pi_ifc_zero, inv_dz)
theta_v_eq = theta_ref.copy()

dw_dt_zero = dw_dt_vertical(theta_v_eq, theta_ref, dpi_dz_zero)
print("=== ТЕСТ 1: опорное состояние ===")
print("dw_dt (должно быть 0):", np.round(dw_dt_zero[:3], 8))
print("Максимум |dw_dt|:", float(np.max(np.abs(dw_dt_zero))))
print()

# ТЕСТ 2: возмущение Exner +1%
exner_test = exner_ref * (1.0 + 0.01 * (1.0 - z_mc / z_mc[-1]))
pi_pr = perturbation(exner_test, exner_ref)
pi_ifc = exner_at_ifc(pi_pr, wgtfac_c)
dpi_dz = vertical_gradient(pi_ifc, inv_dz)

dw_dt = dw_dt_vertical(theta_v_eq, theta_ref, dpi_dz)
print("=== ТЕСТ 2: возмущение Exner +1% у поверхности ===")
print("dw_dt, первые 3 (м/с^2):", np.round(dw_dt[:3], 6))
print("dw_dt, последние 3 (м/с^2):", np.round(dw_dt[-3:], 6))
print()

w1 = step_euler(np.zeros_like(z_mc), dw_dt, 60.0)
print("w после 60 с (м/с), первые 3:", np.round(w1[:3], 4))
print("w после 60 с (м/с), последние 3:", np.round(w1[-3:], 4))
print()

# ТЕСТ 3: тёплый пузырь +2 К на 5 км
theta_v_warm = theta_ref + 2.0 * np.exp(-((z_mc - 5000.0) / 2000.0)**2)
dw_dt_warm = dw_dt_vertical(theta_v_warm, theta_ref, dpi_dz_zero)
print("=== ТЕСТ 3: тёплый пузырь +2 К на 5 км ===")
idx = int(np.argmin(np.abs(z_mc - 5000.0)))
print("dw_dt вокруг 5 км:", np.round(dw_dt_warm[idx-1:idx+2], 4))
print("(ожидаем положительное — тёплый воздух всплывает)")
