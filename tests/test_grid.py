import sys
sys.path.insert(0, "src")
import numpy as np
from grid import make_vertical_grid, interp_to_ifc

grid = make_vertical_grid(nlev=20, z_top=20000.0, stretch=1.15)

print("Число уровней:", len(grid["z_mc"]))
print("Высота первого центра (м):", round(grid["z_mc"][0], 2))
print("Высота последнего центра (м):", round(grid["z_mc"][-1], 2))
print("Толщина нижнего слоя (м):", round(grid["dz"][0], 2))
print("Толщина верхнего слоя (м):", round(grid["dz"][-1], 2))
print("wgtfac_c (первые 5):", np.round(grid["wgtfac_c"][:5], 3))

theta = np.linspace(400.0, 300.0, 20)
theta_ifc = interp_to_ifc(theta, grid["wgtfac_c"])
print("theta на интерфейсах (первые 5):", np.round(theta_ifc[:5], 2))
print("theta на интерфейсах (последние 5):", np.round(theta_ifc[-5:], 2))
