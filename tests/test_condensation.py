import sys
sys.path.insert(0, "src")
import numpy as np
from saturation import (saturation_adjustment, saturation_adjustment_iter,
                        qsat, temperature_from_theta_exner, pressure_from_exner,
                        LV, CPD)

print("=== ТЕСТ 1: сухой воздух (qv < q_sat) — конденсации нет ===")
theta = 300.0
exner = 0.9
qv = 0.001  # 1 г/кг, намного меньше qsat (~18 г/кг)
qc = 0.0
out = saturation_adjustment(theta, qv, qc, exner)
print(f"theta: {out['theta']:.4f}  (было {theta})")
print(f"qv: {out['qv']*1000:.4f} г/кг  (было {qv*1000:.4f})")
print(f"qc: {out['qc']*1000:.4f} г/кг  (было {qc*1000:.4f})")
print(f"dq (сконденсировалось): {out['dq']*1000:.6f} г/кг")
print("Ожидание: ничего не изменилось, dq = 0")

print()
print("=== ТЕСТ 2: пересыщение — конденсация ===")
theta = 300.0
exner = 0.9
qv = 0.025  # 25 г/кг — сильно пересыщено (qsat ~18.7 г/кг)
qc = 0.0
out = saturation_adjustment(theta, qv, qc, exner)
print(f"theta: {out['theta']:.4f} K  (было {theta})")
print(f"qv: {out['qv']*1000:.4f} г/кг  (было {qv*1000:.4f})")
print(f"qc: {out['qc']*1000:.4f} г/кг  (было {qc*1000:.4f})")
print(f"dq (сконденсировалось): {out['dq']*1000:.4f} г/кг")
print()
print("Проверка: qv_new должно быть около qsat:")
T = temperature_from_theta_exner(theta, exner)
p = pressure_from_exner(exner)
qs = qsat(T, p)
print(f"  qsat при исходной T={T:.2f} K, p={p/100:.1f} гПа: {qs*1000:.4f} г/кг")
print(f"  qv_new = {out['qv']*1000:.4f} г/кг  (близко к qsat)")

print()
print("=== ТЕСТ 3: тепловой эффект ===")
dq_val = out['dq']
expected_dtheta = (LV / CPD) * dq_val / exner
print(f"Ожидаемое dtheta = (Lv/cpd) * dq / exner = {expected_dtheta:.4f} K")
print(f"Фактическое dtheta = {out['theta'] - theta:.4f} K")
print(f"Совпадение: {np.isclose(out['theta'] - theta, expected_dtheta)}")
print()
print("Проверка: 1 г/кг конденсации даёт ~2.5 K нагрева (при Lv/cpd=2500, pi=1):")
print(f"  на 1 г/кг: dtheta ≈ {2500 * 0.001:.2f} K")

print()
print("=== ТЕСТ 4: итеративная версия (3 итерации) ===")
out_iter = saturation_adjustment_iter(300.0, 0.025, 0.0, 0.9, n_iter=3)
print(f"theta после итераций: {out_iter['theta']:.4f}")
print(f"qv после итераций:    {out_iter['qv']*1000:.4f} г/кг")
print(f"qc после итераций:    {out_iter['qc']*1000:.4f} г/кг")
print()
print("Ожидание: qv чуть меньше qsat (потому что T выросла, qsat выросло)")

print()
print("=== ТЕСТ 5: сохранение воды ===")
# общая вода = qv + qc должна сохраняться
print(f"Всего воды до:    {(qv + qc)*1000:.4f} г/кг")
print(f"Всего воды после: {(out['qv'] + out['qc'])*1000:.4f} г/кг")
print(f"Изменение: {(out['qv'] + out['qc'] - qv - qc)*1000:.6f} г/кг (должно быть 0)")