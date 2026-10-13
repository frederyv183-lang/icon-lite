"""
Насыщение водяного пара и конденсация.
Упрощённая версия из ICON (mo_nh_moist_thdyn.f90).

Формулы:
    T = theta * pi^(R_d/c_pd)                     (из определения theta)
    p = p0 * pi^(c_pd/R_d)                        (из определения pi)
    e_sat(T) = 610.78 * exp(17.27*(T-273.15) / (T-35.85))   [Pa]  (Тетенс)
    q_sat = eps * e_sat / (p - (1-eps)*e_sat),   eps = 0.622
"""
import numpy as np


P0 = 100000.0        # опорное давление, Па
RD = 287.04          # газовая постоянная сухого воздуха
CPD = 1004.64        # теплоёмкость сухого воздуха при пост. давлении
EPS = 0.622          # R_d / R_v
KAPPA = RD / CPD     # R_d / c_pd ≈ 0.2857
LV = 2.5e6           # удельная теплота парообразования, Дж/кг (const)


def temperature_from_theta_exner(theta, exner):
    """
    Физическая температура из потенциальной и функции Экспера:
        T = theta * pi^(R_d/c_pd)
    """
    return theta * exner ** KAPPA


def pressure_from_exner(exner):
    """
    Давление из функции Экспера:
        p = p0 * pi^(c_pd/R_d)
    """
    return P0 * exner ** (1.0 / KAPPA)


def qsat(T, p):
    """
    Удельная влажность насыщения над водой (формула Тетенса).

    Параметры
    ---------
    T : температура, К
    p : давление, Па

    Возвращает
    ----------
    q_sat : удельная влажность насыщения, кг/кг
    """
    # давление насыщенного пара (Pa)
    es = 610.78 * np.exp(17.27 * (T - 273.15) / (T - 35.85))
    # удельная влажность насыщения
    return EPS * es / (p - (1.0 - EPS) * es)

def saturation_adjustment(theta, qv, qc, exner, lv=LV, cpd=CPD):
    """
    Насыщение-корректировка: конденсация излишка qv в qc с выделением тепла.

    Параметры
    ---------
    theta : (nlev, ...)  потенциальная температура, К
    qv : (nlev, ...)     удельная влажность, кг/кг
    qc : (nlev, ...)     облачная вода, кг/кг
    exner : (nlev, ...)  функция Экспера
    lv, cpd : float      физические константы

    Возвращает
    ----------
    dict с:
        theta : обновлённая потенциальная температура
        qv : обновлённая удельная влажность
        qc : обновлённая облачная вода
    """
    theta = np.asarray(theta, dtype=float).copy()
    qv = np.asarray(qv, dtype=float).copy()
    qc = np.asarray(qc, dtype=float).copy()

    # текущие T и p
    T = temperature_from_theta_exner(theta, exner)
    p = pressure_from_exner(exner)

    # насыщение при текущей T
    qs = qsat(T, p)

    # излишек для конденсации (только где qv > qs)
    dq = np.maximum(qv - qs, 0.0)

    # обновляем qv и qc
    qv_new = qv - dq
    qc_new = qc + dq

    # выделяемое тепло: dT = (Lv/cpd) * dq  (на единицу массы воздуха)
    # В терминах theta: dtheta = dT * (p0/p)^kappa = dT / pi
    dtheta = (lv / cpd) * dq / exner
    theta_new = theta + dtheta

    return {
        "theta": theta_new,
        "qv": qv_new,
        "qc": qc_new,
        "dq": dq,
    }


def saturation_adjustment_iter(theta, qv, qc, exner, n_iter=3):
    """
    Итеративная версия saturation_adjustment — 3 итерации для сходимости.
    Каждая итерация пересчитывает q_sat на обновлённой T и уточняет dq.
    """
    theta = np.asarray(theta, dtype=float).copy()
    qv = np.asarray(qv, dtype=float).copy()
    qc = np.asarray(qc, dtype=float).copy()

    for _ in range(n_iter):
        T = temperature_from_theta_exner(theta, exner)
        p = pressure_from_exner(exner)
        qs = qsat(T, p)
        dq = np.maximum(qv - qs, 0.0)

        # обновляем
        qv = qv - dq
        qc = qc + dq
        dtheta = (LV / CPD) * dq / exner
        theta = theta + dtheta

    return {"theta": theta, "qv": qv, "qc": qc}
