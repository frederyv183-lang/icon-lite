"""
Полный 2D-шаг по времени.
Динамика + перенос + конденсация.
"""
import numpy as np
from state import diagnose_exner
from motion import dw_dt_vertical, dvn_dt_horizontal
from continuity2d import drho_dt_2d
from tracer import step_qv  # работает и для 2D (все операции поэлементные)
from thermo import virtual_theta
from saturation import saturation_adjustment
from precipitation import autoconversion, sedimentation_step, evaporation_step
from exner2d import (perturbation_2d, exner_at_ifc_2d,
                     vertical_gradient_2d, horizontal_gradient_2d)
from grid2d import interp_to_edge, interp_to_center_from_edge
from diffusion2d import km_smagorinsky_2d, diffusion_step_2d


def step_forward_2d(rho, w, vn, theta, qv, qc, qr,
                    exner_ref_2d, theta_ref_2d,
                    grid, dt,
                    qc_crit=5e-4, autoconv_coef=1e-3, Vt=5.0, evap_coef=1e-4,
                    use_diffusion=True):
    """
    Один 2D-шаг по времени.
    """
    wgtfac_c = grid["wgtfac_c"]
    inv_dz = 1.0 / grid["dz"]
    dz = grid["dz"]
    dx = grid["dx"]
    dx_edge = grid["dx_edge"]
    nz = grid["nz"]
    nx = grid["nx"]

    # 1. theta_v (с влажностью)
    theta_v = virtual_theta(theta, qv)

    # 2. Exner
    exner = diagnose_exner(rho, theta_v)

    # 3. Возмущение и градиенты
    pi_pr = perturbation_2d(exner, exner_ref_2d)
    pi_ifc = exner_at_ifc_2d(pi_pr, wgtfac_c)
    dpi_dz = vertical_gradient_2d(pi_ifc, inv_dz)        # (nz, nx)
    dpi_dx = horizontal_gradient_2d(pi_pr, dx_edge)      # (nz, nx+1)

    # 4. Тенденция w (вертикальная динамика)
    dw_dt_mc = dw_dt_vertical(theta_v, theta_ref_2d, dpi_dz)  # (nz, nx)

    # интерполяция на горизонтальные интерфейсы
    dw_dt_ifc = np.zeros((nz + 1, nx))
    dw_dt_ifc[1:-1, :] = 0.5 * (dw_dt_mc[:-1, :] + dw_dt_mc[1:, :])
    dw_dt_ifc[0, :] = dw_dt_mc[0, :]
    dw_dt_ifc[-1, :] = dw_dt_mc[-1, :]

    # 5. Тенденция vn (горизонтальная динамика)
    theta_v_edge = interp_to_edge(theta_v)
    dvn_dt = dvn_dt_horizontal(theta_v_edge, dpi_dx)

    # 6. Обновление скоростей
    w_new = w + dt * dw_dt_ifc
    w_new[0, :] = 0.0
    w_new[-1, :] = 0.0

    vn_new = vn + dt * dvn_dt
    vn_new[:, 0] = 0.0
    vn_new[:, -1] = 0.0

    # 6b. Турбулентная диффузия скалярных полей (Смагоринский + Ri)
    if use_diffusion:
        # Строим dz_ifc и dx_ifc (расстояния между центрами)
        dz_ifc = np.zeros(nz + 1)
        dz_ifc[1:-1] = 0.5 * (dz[:-1] + dz[1:])
        dz_ifc[0] = dz[0]
        dz_ifc[-1] = dz[-1]

        dx_ifc = np.zeros(nx + 1)
        dx_ifc[1:-1] = 0.5 * (dx[:-1] + dx[1:])
        dx_ifc[0] = dx[0]
        dx_ifc[-1] = dx[-1]

        # К_m по актуальным полям
        theta_v_dyn = virtual_theta(theta, qv)
        Km = km_smagorinsky_2d(theta_v_dyn, w_new, vn_new,
                               inv_dz, dz, dz_ifc, dx, dx_ifc)

        # Применяем к скалярным полям
        theta = diffusion_step_2d(theta, Km, dz, dz_ifc, dx, dx_ifc, dt)
        qv = diffusion_step_2d(qv, Km, dz, dz_ifc, dx, dx_ifc, dt)
        qc = diffusion_step_2d(qc, Km, dz, dz_ifc, dx, dx_ifc, dt)
        qr = diffusion_step_2d(qr, Km, dz, dz_ifc, dx, dx_ifc, dt)

        # Защита от отрицательных значений
        qv = np.maximum(qv, 0.0)
        qc = np.maximum(qc, 0.0)
        qr = np.maximum(qr, 0.0)

    # 7. Обновление плотности (2D неразрывность)
    drho = drho_dt_2d(rho, w_new, vn_new, dx, inv_dz)
    rho_new = rho + dt * drho

    # 8. Перенос qv (используем вертикальный поток для простоты)
    qv_advected = step_qv(rho, qv, w_new, inv_dz, dt)

    # 9. Конденсация
    theta_v_new = virtual_theta(theta, qv_advected)
    exner_new = diagnose_exner(rho_new, theta_v_new)
    sat = saturation_adjustment(theta, qv_advected, qc, exner_new)

    theta_new = sat["theta"]
    qv_new = sat["qv"]
    qc_new = sat["qc"]

    # --- Осадки (qr) ---
    # 1. Автоконверсия qc -> qr
    ac = autoconversion(qc_new, dt, qc_crit=qc_crit, coef=autoconv_coef)
    qc_new = ac["qc"]
    qr_new = qr + ac["qr"]

    # 2. Осаждение (qr падает)
    qr_new, precip_flux = sedimentation_step(qr_new, Vt, dz, dt)

    # 3. Испарение (часть qr -> qv)
    ev = evaporation_step(qr_new, qv_new, qc_new, exner_new, dt, evap_coef=evap_coef)
    qr_new = ev["qr"]
    qv_new = ev["qv"]

    # 10. Финальные диагностики
    theta_v_final = virtual_theta(theta_new, qv_new)
    exner_final = diagnose_exner(rho_new, theta_v_final)

    return {
        "rho": rho_new,
        "w": w_new,
        "vn": vn_new,
        "theta": theta_new,
        "qv": qv_new,
        "qc": qc_new,
        "qr": qr_new,
        "precip_flux": precip_flux,
        "theta_v": theta_v_final,
        "exner": exner_final,
        "pi_pr": perturbation_2d(exner_final, exner_ref_2d),
    }