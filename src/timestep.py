"""
Полный шаг по времени (упрощённый, явная схема Эйлера).
Связывает движение -> плотность -> Exner.
"""
import numpy as np
from state import CPD, G, diagnose_exner
from exner import perturbation, exner_at_ifc, vertical_gradient
from motion import dw_dt_vertical
from continuity import drho_dt


def step_forward(rho, w, theta_v, exner_ref, theta_ref, wgtfac_c, inv_dz, dt):
    """
    Один шаг по времени.
    """
    # 1. Диагностика Exner из текущих rho, theta_v
    exner = diagnose_exner(rho, theta_v)

    # 2. Возмущение
    pi_pr = perturbation(exner, exner_ref)

    # 3. Интерполяция на интерфейсы + градиент
    pi_ifc = exner_at_ifc(pi_pr, wgtfac_c)
    dpi_dz = vertical_gradient(pi_ifc, inv_dz)   # размер (nlev,)

    # 4. Тенденция w в центрах ячеек
    dw_dt_mc = dw_dt_vertical(theta_v, theta_ref, dpi_dz)  # (nlev,)

    # 5. Интерполяция тенденции на интерфейсы (nlev -> nlev+1)
    dw_dt_ifc = np.zeros(len(w))
    dw_dt_ifc[1:-1] = 0.5 * (dw_dt_mc[:-1] + dw_dt_mc[1:])
    dw_dt_ifc[0] = dw_dt_mc[0]
    dw_dt_ifc[-1] = dw_dt_mc[-1]

    # 6. Обновление w
    w_new = w + dt * dw_dt_ifc
    # граничные условия: w=0 на верхней и нижней границе (жёсткая крышка)
    w_new[0] = 0.0
    w_new[-1] = 0.0

    # 7. Тенденция rho
    drho = drho_dt(rho, w_new, inv_dz)

    # 8. Обновление rho
    rho_new = rho + dt * drho

    # 9. Диагностика нового Exner
    exner_new = diagnose_exner(rho_new, theta_v)

    return {
        "rho": rho_new,
        "w": w_new,
        "exner": exner_new,
        "pi_pr": perturbation(exner_new, exner_ref),
    }


def run(rho0, w0, theta_v, exner_ref, theta_ref, wgtfac_c, inv_dz,
        dt=1.0, nsteps=100, verbose_every=20):
    """Прогон модели на nsteps шагов."""
    rho = rho0.copy()
    w = w0.copy()
    history = []

    for n in range(nsteps):
        result = step_forward(rho, w, theta_v, exner_ref, theta_ref,
                              wgtfac_c, inv_dz, dt)
        rho = result["rho"]
        w = result["w"]

        if n % verbose_every == 0:
            print(f"  шаг {n:4d}: max|w| = {np.max(np.abs(w)):.4e} м/с, "
                  f"max|pi'| = {np.max(np.abs(result['pi_pr'])):.4e}")

        history.append({
            "t": (n + 1) * dt,
            "rho": rho.copy(),
            "w": w.copy(),
            "exner": result["exner"].copy(),
            "pi_pr": result["pi_pr"].copy(),
        })

    return history
