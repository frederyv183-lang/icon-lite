import sys
sys.path.insert(0, "src")
import numpy as np
from grid import make_vertical_grid
from state import make_reference_atmosphere
from timestep import run

grid = make_vertical_grid(nlev=20, z_top=20000.0, stretch=1.15)
z_mc = grid["z_mc"]
z_ifc = grid["z_ifc"]
wgtfac_c = grid["wgtfac_c"]
inv_dz = 1.0 / grid["dz"]

ref = make_reference_atmosphere(z_mc, z_ifc, theta_surf=300.0)
rho0 = ref["rho_ref"].copy()
theta_ref = ref["theta_ref"]
exner_ref = ref["exner_ref"]

# Начальное возмущение: тёплый пузырь +2 К на 5 км
theta_v = theta_ref + 2.0 * np.exp(-((z_mc - 5000.0) / 2000.0)**2)

# w = 0 в начальный момент
w0 = np.zeros(len(z_ifc))

print("=== Прогон на 100 шагов, dt=1 с ===")
history = run(rho0, w0, theta_v, exner_ref, theta_ref,
              wgtfac_c, inv_dz, dt=1.0, nsteps=100, verbose_every=20)

print()
print("=== Финальное состояние ===")
final = history[-1]
print("t =", final["t"], "с")
print("max|w| =", float(np.max(np.abs(final["w"]))), "м/с")
print("max|rho - rho0| =", float(np.max(np.abs(final["rho"] - rho0))), "кг/м3")
print("max|pi'| =", float(np.max(np.abs(final["pi_pr"]))))

# Проверим, что тёплый пузырь дал положительное w вокруг 5 км
idx = int(np.argmin(np.abs(z_ifc - 5000.0)))
print()
print("=== Ожидание: воздух в пузыре поднимается ===")
print("w вокруг 5 км:", np.round(final["w"][idx-1:idx+3], 4), "м/с")
print("Знак положительный:", bool(final["w"][idx] > 0))
