import sys
sys.path.insert(0, "src")
import numpy as np
from grid import make_vertical_grid
from state import make_reference_atmosphere
from diffusion import diffusion_step

grid = make_vertical_grid(nlev=40, z_top=20000.0, stretch=1.15)
z_mc = grid["z_mc"]
z_ifc = grid["z_ifc"]
dz = grid["dz"]

# --- ТЕСТ 1: однородное поле -> диффузия ничего не делает ---
field_const = np.ones(40) * 5.0
Km_const = np.ones(40) * 10.0
out = diffusion_step(field_const, Km_const, z_mc, z_ifc, dz, dt=60.0)
print("=== ТЕСТ 1: однородное поле ===")
print("max|out - field| =", float(np.max(np.abs(out - field_const))))
print("Ожидание: 0")
print()

# --- ТЕСТ 2: узкий пик -> диффузия размывает ---
field = np.zeros(40)
field[15:25] = 1.0  # ступенька
out = diffusion_step(field, Km_const, z_mc, z_ifc, dz, dt=600.0)
print("=== ТЕСТ 2: размывание ступеньки за 600 с ===")
print("Значения около границы (k=13..16):")
print("  было:", field[13:17])
print("  стало:", np.round(out[13:17], 4))
print("Ожидание: значения около границы слегка уменьшились,")
print("          вне ступеньки появились малые значения > 0")
print()

# --- ТЕСТ 3: сохранение интеграла ---
mass_before = float(np.sum(field * dz))
mass_after = float(np.sum(out * dz))
print("=== ТЕСТ 3: сохранение интеграла ===")
print(f"Интеграл до:    {mass_before:.6f}")
print(f"Интеграл после: {mass_after:.6f}")
print(f"Изменение: {100*(mass_after/mass_before - 1):.4f}%")
print("Ожидание: < 0.1%")
print()

# --- ТЕСТ 4: диффузия уменьшает дисперсию ---
print("=== ТЕСТ 4: уменьшение дисперсии ===")
print(f"Max field до:  {field.max():.4f}")
print(f"Max field после: {out.max():.4f}")
print(f"Отношение: {out.max()/field.max():.4f}")
print("Ожидание: < 1 (пик сгладился)")