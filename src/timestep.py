"""
Полный шаг по времени с учётом влажности.
"""
import numpy as np
from state import CPD, G, diagnose_exner
from exner import perturbation, exner_at_ifc, vertical_gradient
from motion import dw_dt_vertical
from continuity import drho_dt
from tracer import step_qv
from thermo import virtual_theta


def step_forward(rho, w, theta, qv, exner_ref, theta_ref,
                 wgtfac_c, inv_dz, dt):
    """
    Один шаг по времени с влажностью.
    """
    # 1. Виртуальная потенциальная температура (с учётом влажности)
    theta_v = virtual_theta(theta, qv)

    # 2. Диагностика Exner из rho, theta_v
    exner = diagnose_exner(rho, theta_v)

    # 3. Возмущение Exner
    pi_pr = perturbation(exner, exner_ref)

    # 4. Градиент Exner
    pi_ifc = exner_at_ifc(pi_pr, wgtfac_c)
    dpi_dz = vertical_gradient(pi_ifc, inv_dz)

    # 5. Тенденция w (используем виртуальную theta_v и опорную theta_ref)
    dw_dt_mc = dw_dt_vertical(theta_v, theta_ref, dpi_dz)

    # 6. Интерполяция тенденции на интерфейсы
    dw_dt_ifc = np.zeros(len(w))
    dw_dt_ifc[1:-1] = 0.5 * (dw_dt_mc[:-1] + dw_dt_mc[1:])
    dw_dt_ifc[0] = dw_dt_mc[0]
    dw_dt_ifc[-1] = dw_dt_mc[-1]

    # 7. Обновление w
    w_new = w + dt * dw_dt_ifc
    w_new[0] = 0.0
    w_new[-1] = 0.0

    # 8. Обновление rho
    drho = drho_dt(rho, w_new, inv_dz)
    rho_new = rho + dt * drho

    # 9. Обновление qv
    qv_new = step_qv(rho, qv, w_new, inv_dz, dt)

    # 10. Диагностика нового Exner
    theta_v_new = virtual_theta(theta, qv_new)
    exner_new = diagnose_exner(rho_new, theta_v_new)

    return {
        "rho": rho_new,
        "w": w_new,
        "qv": qv_new,
        "theta_v": theta_v_new,
        "exner": exner_new,
        "pi_pr": perturbation(exner_new, exner_ref),
    }