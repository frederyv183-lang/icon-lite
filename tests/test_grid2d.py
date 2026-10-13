import sys
sys.path.insert(0, "src")
import numpy as np
from grid2d import (make_grid2d, interp_to_edge, interp_to_center_from_edge)

g = make_grid2d(nz=40, nx=80, x_total=40000.0, z_top=20000.0)

print("=== Размеры сетки ===")
print(f"nz = {g['nz']}, nx = {g['nx']}")
print(f"Форма z_mc: {g['z_mc'].shape}, dz: {g['dz'].shape}")
print(f"Форма x_mc: {g['x_mc'].shape}, x_edge: {g['x_edge'].shape}")
print(f"Форма dx_edge: {g['dx_edge'].shape}")
print()

print("=== Проверка координат x ===")
print(f"x_edge[:5] = {np.round(g['x_edge'][:5], 1)} м")
print(f"x_mc[:5] = {np.round(g['x_mc'][:5], 1)} м")
print(f"dx = {g['dx'][0]:.1f} м (постоянна: {np.allclose(g['dx'], g['dx'][0])})")
print()

print("=== Проверка dx_edge ===")
print(f"dx_edge[1] = {g['dx_edge'][1]:.1f} м  (должна быть 500 при 80 ячейках в 40 км)")
print(f"dx_edge[0] = {g['dx_edge'][0]:.1f} м  (копия от dx_edge[1])")
print()

print("=== Интерполяция центр → ребро ===")
# сделаем тестовое поле — синусоиду
x = g["x_mc"]
field_mc = np.zeros((g["nz"], g["nx"]))
field_mc[10, :] = np.sin(2 * np.pi * x / 10000.0)  # длина волны 10 км
field_edge = interp_to_edge(field_mc)
print(f"Форма field_mc:   {field_mc.shape}")
print(f"Форма field_edge: {field_edge.shape}")
print(f"field_mc[10, 5:8] = {np.round(field_mc[10, 5:8], 4)}")
print(f"field_edge[10, 5:9] = {np.round(field_edge[10, 5:9], 4)}")
print("Ожидание: edge[6] = среднее от mc[5] и mc[6]")
print()

print("=== Интерполяция ребро → центр ===")
field_back = interp_to_center_from_edge(field_edge)
print(f"Максимальная разница (mc → edge → mc): "
      f"{float(np.max(np.abs(field_back - field_mc))):.6f}")
print("Ожидание: ~0 (интерполяция симметрична для гладких полей)")
print()

print("=== Проверка формы при разных nx ===")
for nx_test in [40, 80, 160, 320]:
    gt = make_grid2d(nz=20, nx=nx_test, x_total=40000.0)
    dx = gt['dx'][0]
    print(f"nx = {nx_test:4d}: dx = {dx:6.1f} м, dx_edge[1] = {gt['dx_edge'][1]:6.1f} м")