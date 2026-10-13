"""Осадки: автоконверсия qc -> qr, осаждение, испарение.

Упрощённая схема, вдохновлённая ICON (mo_nh_moist_thdyn.f90,
mo_solve_nonhydro.f90, блок precipitation).

Обозначения:
    qc : облачная вода (kg/kg)
    qr : дождевая вода (kg/kg)
    qv : водяной пар (kg/kg)

Процессы:
1. autoconversion: qc > qc_crit -> часть qc переходит в qr.
2. sedimentation: qr падает со скоростью Vt (м/с).
3. evaporation: qr испаряется в qv при ненасыщении.
"""
import numpy as np


def autoconversion(qc, dt, qc_crit=5e-4, coef=1e-3):
    """
    Автоконверсия облачной воды в дождевую.

    Формула (упрощённый Kessler):
        dqr = coef * max(qc - qc_crit, 0) * dt
        qc_new = qc - dqr

    Параметры
    ---------
    qc : (nlev, ...)   облачная вода, кг/кг
    dt : float         шаг по времени, с
    qc_crit : float    порог автоконверсии (по умолчанию 0.5 г/кг)
    coef : float       коэффициент скорости (1/с)

    Возвращает
    ----------
    dict с:
        qc : обновлённая облачная вода
        qr : прирост дождевой воды (dqr)
    """
    excess = np.maximum(qc - qc_crit, 0.0)
    dqr = coef * excess * dt
    # не больше, чем есть облачной воды
    dqr = np.minimum(dqr, qc)
    qc_new = qc - dqr
    return {"qc": qc_new, "qr": dqr}


def sedimentation_step(qr, Vt, dz, dt):
    """
    Осаждение дождевой воды: qr падает со скоростью Vt.

    Простая явная схема upwind:
        F_ifc[k] = Vt * qr[k]      (поток вниз через верхнюю границу ячейки k)
        qr_new[k] = qr[k] + dt * (F_ifc[k-1] - F_ifc[k]) / dz[k]

    Здесь k=0 — поверхность, k=nlev-1 — верх.

    Параметры
    ---------
    qr : (nlev, ...)   дождевая вода, кг/кг
    Vt : float         скорость падения, м/с (положительная, вниз)
    dz : (nlev,)       толщины слоёв, м
    dt : float         шаг по времени, с

    Возвращает
    ----------
    qr_new : (nlev, ...)
    """
    nlev = qr.shape[0]
    qr_new = qr.copy()

    # dz для broadcast
    if dz.ndim == 1 and qr.ndim > 1:
        dz_b = dz.reshape((nlev,) + (1,) * (qr.ndim - 1))
    else:
        dz_b = dz

    # CFL-проверка: Vt*dt/dz <= 1
    cfl = Vt * dt / dz.min()
    if cfl > 1.0:
        # ограничиваем шаг, чтобы схема была устойчива
        # (просто уменьшаем эффективный Vt)
        Vt_eff = 0.9 * dz.min() / dt
    else:
        Vt_eff = Vt

    # поток вниз через интерфейсы: F[k] — поток из ячейки k в k-1
    # F_ifc[k] = Vt * qr[k] — масса, покидающая ячейку k вниз за единицу времени
    F = Vt_eff * qr                     # (nlev, ...)

    # приток снизу: из ячейки k+1 в k
    # отток: из ячейки k в k-1
    # dqr[k]/dt = (F[k+1] - F[k]) / dz[k]   (при k+1 < nlev)
    # На верхней границе F[nlev] = 0 (нет источника выше)
    # На нижней границе F[-1] уходит из модели (QR уходит в осадки)

    dqr = np.zeros_like(qr)
    # для k=0..nlev-2: приток от k+1, отток в k-1
    dqr[0:-1] = (F[1:] - F[0:-1]) / dz_b[0:-1]
    # для k=nlev-1: притока сверху нет, только отток
    dqr[-1] = -F[-1] / dz_b[-1]

    qr_new = qr + dt * dqr
    qr_new = np.maximum(qr_new, 0.0)

    # масса, выпавшая на поверхность = F[0] * dt (уходит из модели)
    precip_flux = F[0]  # кг/(кг·с) * ... — приведённая величина

    return qr_new, precip_flux


def evaporation_step(qr, qv, qc, exner, dt, evap_coef=1e-4, lv=2.5e6, cpd=1004.64):
    """
    Испарение дождевой воды в ненасыщенном воздухе.

    Упрощённо: если qv < qsat(T,p), часть qr испаряется:
        dqv_evap = evap_coef * (qsat - qv) * dt
        dqr_evap = -dqv_evap

    Параметры
    ---------
    qr : (nlev, ...)   дождевая вода, кг/кг
    qv : (nlev, ...)   водяной пар, кг/кг
    qc : (nlev, ...)   облачная вода (для вычисления exner, если нужно)
    exner : (nlev, ...)
    dt : float
    evap_coef : float  скорость испарения
    lv, cpd : float    физические константы

    Возвращает
    ----------
    dict с:
        qr : обновлённая дождевая вода
        qv : обновлённый водяной пар
        qc : без изменений
    """
    from saturation import qsat, temperature_from_theta_exner, pressure_from_exner

    # Для расчёта qsat нужна theta. Но у нас qv,qc,exner. Используем приближение:
    # T = theta * exner^kappa, theta берём из... к сожалению, theta недоступна.
    # Поэтому используем простую параметризацию: испарение только если qv < qv_sat_threshold.
    # Здесь упрощаем: qv_sat_approx ~ 0.015 (15 г/кг) при обычных условиях.
    # Это грубое приближение, но достаточное для демонстрации.

    qv_sat_approx = 0.015  # кг/кг

    dq_evap = evap_coef * np.maximum(qv_sat_approx - qv, 0.0) * dt
    dq_evap = np.minimum(dq_evap, qr)

    qr_new = qr - dq_evap
    qv_new = qv + dq_evap

    return {"qr": qr_new, "qv": qv_new, "qc": qc}