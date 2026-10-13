import sys
sys.path.insert(0, "src")
import numpy as np
from grid import make_vertical_grid
from state import make_reference_atmosphere
from diffusion import km_smagorinsky, deformation_vertical, richardson_number

grid = make_vertical_grid(nlev=40, z_top=20000.0, stretch=1.15)
z_mc = grid["z_mc"]
z_ifc = grid["z_ifc"]
inv_dz = 1.0 / grid["dz"]
dz = grid["dz"]

ref = make_reference_atmosphere(z_mc, z_ifc, theta_surf=300.0)
theta_v = ref["theta_ref"].copy()

# --- ТЕСТ 1: покой (w=0) -> |S|=0 -> Km минимальный ---
w_zero = np.zeros(len(z_ifc))
S_zero = deformation_vertical(w_zero, inv_dz)
Km_zero = km_smagorinsky(theta_v, w_zero, z_mc, z_ifc, inv_dz, dz)
print("=== ТЕСТ 1: покой w=0 ===")
print("max|S| =", float(np.max(np.abs(S_zero))))
print("max Km =", float(np.max(Km_zero)), "м^2/с")
print("Ожидание: S=0, Km ~ K_MIN (численный пол)")
print()

# --- ТЕСТ 2: сильный сдвиг w нарастает с высотой ---
w_shear = np.linspace(0.0, 10.0, len(z_ifc))  # 0 -> 10 м/с
S_shear = deformation_vertical(w_shear, inv_dz)
Km_shear = km_smagorinsky(theta_v, w_shear, z_mc, z_ifc, inv_dz, dz)
print("=== ТЕСТ 2: сдвиг w = 0 -> 10 м/с ===")
print("max|S| =", float(np.max(np.abs(S_shear))), "1/с")
print("max Km =", float(np.max(Km_shear)), "м^2/с")
print("Ожидание: |S| ~ 0.001-0.01 1/с, Km ~ 1-100 м^2/с")
print()

# --- ТЕСТ 3: устойчивая стратификация подавляет Km ---
# theta растёт с высотой -> стратификация устойчивая -> Ri > 0 -> F < 1
theta_stable = ref["theta_ref"] + 20.0 * (z_mc / z_mc[-1])  # +20 K к верху
Km_stable = km_smagorinsky(theta_stable, w_shear, z_mc, z_ifc, inv_dz, dz)
Ri_stable = richardson_number(theta_stable, w_shear, inv_dz, dz)
print("=== ТЕСТ 3: устойчивая стратификация ===")
print("Ri в середине:", float(Ri_stable[20]))
print("max Km =", float(np.max(Km_stable)), "м^2/с")
print("Ожидание: Km меньше, чем без стратификации")
print("Отношение Km_stable / Km_neutral:",
      float(np.max(Km_stable) / np.max(Km_shear)))
print()

# --- ТЕСТ 4: неустойчивая стратификация ---
theta_unstable = ref["theta_ref"] - 10.0 * (z_mc / z_mc[-1])
Km_unstable = km_smagorinsky(theta_unstable, w_shear, z_mc, z_ifc, inv_dz, dz)
print("=== ТЕСТ 4: неустойчивая стратификация ===")
print("max Km =", float(np.max(Km_unstable)), "м^2/с")
print("Отношение к нейтральному:", float(np.max(Km_unstable) / np.max(Km_shear)))
print("Ожидание: Km такая же или больше (Ri < 0 -> F = 1)")
print()

# --- ТЕСТ 5: профиль Km по высоте ---
print("=== ТЕСТ 5: профиль Km ===")
for k in [0, 10, 20, 30, 39]:
    print(f"  k={k:2d}, z={z_mc[k]:6.0f} м: Km = {Km_shear[k]:8.3f} м^2/с")