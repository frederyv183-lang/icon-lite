import sys
sys.path.insert(0, "src")
import numpy as np
from thermo import virtual_theta, virtual_theta_approx, RV_OVER_RD

theta = 300.0  # К

print("=== theta_v для разных qv ===")
for qv in [0.0, 0.001, 0.005, 0.010, 0.020]:
    tv = virtual_theta(theta, qv)
    tv_approx = virtual_theta_approx(theta, qv)
    print(f"qv = {qv:.3f}: theta_v = {tv:.3f} К, approx = {tv_approx:.3f} К, "
          f"разница = {abs(tv - tv_approx):.4f} К")

print()
print("=== Проверка: сухой воздух (qv=0) ===")
tv0 = virtual_theta(theta, 0.0)
print("theta_v при qv=0:", tv0, "| должно быть 300.0:", tv0 == theta)

print()
print("=== Плавучесть от влажности ===")
theta_ref = 300.0
# тёплый сухой пузырь +2 К
buoy_dry = 9.80665 * (virtual_theta(theta_ref + 2.0, 0.0) - theta_ref) / theta_ref
# тот же пузырь, но влажный (qv = 10 г/кг)
buoy_moist = 9.80665 * (virtual_theta(theta_ref + 2.0, 0.010) - theta_ref) / theta_ref
print(f"Плавучесть сухого пузыря +2 К: {buoy_dry:.4f} м/с²")
print(f"Плавучесть влажного пузыря +2 К, qv=10 г/кг: {buoy_moist:.4f} м/с²")
print(f"Прирост: {100*(buoy_moist/buoy_dry - 1):.1f}%")