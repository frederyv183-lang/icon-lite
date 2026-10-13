import sys
sys.path.insert(0, "src")
import numpy as np
from grid2d import make_grid2d, interp_to_edge
from state2d import make_initial_2d
from exner2d import (perturbation_2d, exner_at_ifc_2d,
                     vertical_gradient_2d, horizontal_gradient_2d)
from state import diagnose_exner
from thermo import virtual_theta

g = make_grid2d(nz=40, nx=80, x_total=40000.0, z_top=20000.0)
init = make_initial_2d(g, None, qv_max=0.020, dtheta=2.0,
                       x0_km=20.0, z0_km=3.0, radius_x_km=5.0, radius_z_km=1.5)

# Exner из состояния (диагностика)
theta_v = virtual_theta(init["theta"], init["qv"])
exner = diagnose_exner(init["rho"], theta_v)

# опорный Exner — из опорной theta и rho
theta_v_ref = virtual_theta(init["theta_ref_2d"], np.zeros_like(init["qv"]))
from state import diagnose_exner as dx
exner_ref = dx(init["rho"], theta_v_ref)

# возмущение Exner
pi_pr = perturbation_2d(exner, exner_ref)
print("=== Возмущение Exner pi' ===")
print(f"Форма: {pi_pr.shape}")
i0 = 39
k0 = 7
print(f"pi' в центре пузыря: {pi_pr[k0, i0]:.6f}")
print(f"pi' на краю области: {pi_pr[k0, 0]:.6f}")
print()

# горизонтальный градиент
dpi_dx = horizontal_gradient_2d(pi_pr, g["dx_edge"])
print("=== Горизонтальный градиент dpi'/dx ===")
print(f"Форма: {dpi_dx.shape}")
# градиент должен быть положительный слева от центра, отрицательный справа
print(f"dpi/dx[k={k0}, edge=38]: {dpi_dx[k0, 38]:.6e}")
print(f"dpi/dx[k={k0}, edge=39]: {dpi_dx[k0, 39]:.6e}")
print(f"dpi/dx[k={k0}, edge=40]: {dpi_dx[k0, 40]:.6e}")
print(f"dpi/dx[k={k0}, edge=41]: {dpi_dx[k0, 41]:.6e}")
print()
print("Ожидание: слева от центра (i<39) положительный,")
print("          справа (i>40) отрицательный — пузырь давит в стороны.")
print()

# тенденция vn
theta_v_edge = interp_to_edge(theta_v)
from motion import dvn_dt_horizontal
dvn = dvn_dt_horizontal(theta_v_edge, dpi_dx)
print("=== Тенденция vn ===")
print(f"Форма: {dvn.shape}")
print(f"dvn/dt[k={k0}, edge=38]: {dvn[k0, 38]:.6e} м/с²")
print(f"dvn/dt[k={k0}, edge=40]: {dvn[k0, 40]:.6e} м/с²")
print()
print("Ожидание: слева — vn ускоряется влево (отрицательное),")
print("          справа — vn ускоряется вправо (положительное).")
print("          Знак = минус градиент (градиент давит ИЗ высокого p).")