"""
Полный шаг по времени с учётом влажности И конденсации.
"""
import numpy as np
from state import CPD, G, diagnose_exner
from exner import perturbation, exner_at_ifc, vertical_gradient
from motion import dw_dt_vertical
from continuity import drho_dt
from tracer import step_qv
from thermo import virtual_theta
from saturation import saturation_adjustment


def step_forward(rho, w, theta, qv, qc, exner_ref, theta_ref,
                 wgtfac_c, inv_dz, dt):
    """
    Один шаг по времени: динамика + перенос влаги + конденсация.

    Параметры
    ---------
    rho, w, theta, qv, qc, exner_ref, theta_ref, wgtfac_c, inv_dz : как раньше
    qc : (nlev,) облачная вода, кг/кг
    """
    # 1. Виртуальная theta с учётом влажности и облачности
    #    (облачная вода добавляет ещё немного плавучести — qc тоже вычитается
    #     из «сухой» массы, но эффект мал; в ICON учитывается через theta_v)
    theta_v = virtual_theta(theta, qv)

    # 2. Exner из состояния
    exner = diagnose_exner(rho, theta_v)

    # 3. Возмущение Exner и градиент
    pi_pr = perturbation(exner, exner_ref)
    pi_ifc = exner_at_ifc(pi_pr, wgtfac_c)
    dpi_dz = vertical_gradient(pi_ifc, inv_dz)

    # 4. Тенденция w и шаг
    dw_dt_mc = dw_dt_vertical(theta_v, theta_ref, dpi_dz)
    dw_dt_ifc = np.zeros(len(w))
    dw_dt_ifc[1:-1] = 0.5 * (dw_dt_mc[:-1] + dw_dt_mc[1:])
    dw_dt_ifc[0] = dw_dt_mc[0]
    dw_dt_ifc[-1] = dw_dt_mc[-1]

    w_new = w + dt * dw_dt_ifc
    w_new[0] = 0.0
    w_new[-1] = 0.0

    # 5. Перенос rho и qv (без конденсации)
    drho = drho_dt(rho, w_new, inv_dz)
    rho_new = rho + dt * drho
    qv_advected = step_qv(rho, qv, w_new, inv_dz, dt)

    # 6. Конденсация: подтягиваем θ, qv, qc к равновесному состоянию
    #    используем Exner, посчитанный из текущего rho_new, theta_v
    theta_v_new = virtual_theta(theta, qv_advected)
    exner_new = diagnose_exner(rho_new, theta_v_new)

    sat = saturation_adjustment(theta, qv_advected, qc, exner_new)
    theta_new = sat["theta"]
    qv_new = sat["qv"]
    qc_new = sat["qc"]

    # 7. Финальное состояние
    theta_v_final = virtual_theta(theta_new, qv_new)
    exner_final = diagnose_exner(rho_new, theta_v_final)

    return {
        "rho": rho_new,
        "w": w_new,
        "theta": theta_new,
        "qv": qv_new,
        "qc": qc_new,
        "theta_v": theta_v_final,
        "exner": exner_final,
        "pi_pr": perturbation(exner_final, exner_ref),
    }