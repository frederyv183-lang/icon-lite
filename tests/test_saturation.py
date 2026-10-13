import sys
sys.path.insert(0, "src")
import numpy as np
from saturation import (temperature_from_theta_exner, pressure_from_exner,
                        qsat, KAPPA, P0)

print("=== ТЕСТ 1: T из theta, exner ===")
# theta = 300 K, exner = 0.9 -> T = 300 * 0.9^0.2857
T = temperature_from_theta_exner(300.0, 0.9)
print(f"theta=300, pi=0.9 -> T = {T:.3f} K")
# ожидаем около 300*0.9^0.2857 = 300*0.9703 = 291.1 K

print()
print("=== ТЕСТ 2: давление из exner ===")
p = pressure_from_exner(0.9)
print(f"pi=0.9 -> p = {p:.0f} Па")
# ожидаем 100000 * 0.9^3.5 = 100000 * 0.6964 = 69640 Па

print()
print("=== ТЕСТ 3: qsat при разных условиях ===")
cases = [
    (273.15, 100000.0, "T=0 C, p=1000 hPa"),
    (283.15, 100000.0, "T=10 C, p=1000 hPa"),
    (293.15, 100000.0, "T=20 C, p=1000 hPa"),
    (303.15, 100000.0, "T=30 C, p=1000 hPa"),
    (293.15, 85000.0, "T=20 C, p=850 hPa"),
    (293.15, 50000.0, "T=20 C, p=500 hPa"),
]
print(f"{'условие':<25} {'qsat (г/кг)':>12}")
for T, p, label in cases:
    qs = qsat(T, p)
    print(f"{label:<25} {qs*1000:>12.3f}")
print()
print("Ожидание: qsat растёт с T, падает с высотой (с p):")
print("  T=0 C: ~3.8 г/кг")
print("  T=20 C: ~14.9 г/кг")
print("  T=30 C: ~27.7 г/кг")
print("  T=20 C, p=500 гПа: ~29 г/кг (высота → меньше p → больше qsat)")

print()
print("=== ТЕСТ 4: типичный профиль атмосферы ===")
theta = 300.0
for exner in [1.0, 0.9, 0.8, 0.7, 0.5, 0.3]:
    T = temperature_from_theta_exner(theta, exner)
    p = pressure_from_exner(exner)
    qs = qsat(T, p)
    print(f"pi={exner:.2f}  T={T:6.1f} K ({T-273.15:5.1f} C)  p={p/100:6.1f} hPa  qsat={qs*1000:6.2f} г/кг")