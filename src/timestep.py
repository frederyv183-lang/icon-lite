"""
Полный шаг по времени: динамика + влага + конденсация + турбулентность.
"""
import numpy as np
from state import CPD, G, diagnose_exner
from exner import perturbation, exner_at_ifc, vertical_gradient
from motion import dw_dt_vertical
from continuity import drho_dt
from tracer import step_qv
from fog import radiative_cooling_step
from thermo import virtual_theta
from saturation import saturation_adjustment
from precipitation import autoconversion, sedimentation_step, evaporation_step
from diffusion import km_smagorinsky, diffusion_step


def step_forward(rho, w, theta, qv, qc, qr, exner_ref, theta_ref,
                 wgtfac_c, inv_dz, dz, z_mc, z_ifc, dt,
                 use_diffusion=True, dx=500.0,
                 qc_crit=5e-4, autoconv_coef=1e-3, Vt=5.0, evap_coef=1e-4,
                 radiative_cooling=False, k_rad=1e-4):
    """Один шаг по времени с турбулентной диффузией."""
    if radiative_cooling:
        theta = radiative_cooling_step(theta, dt, k_rad=k_rad, theta_ref=theta_ref)

    theta_v = virtual_theta(theta, qv)
    exner = diagnose_exner(rho, theta_v)
    pi_pr = perturbation(exner, exner_ref)
    pi_ifc = exner_at_ifc(pi_pr, wgtfac_c)
    dpi_dz = vertical_gradient(pi_ifc, inv_dz)

    dw_dt_mc = dw_dt_vertical(theta_v, theta_ref, dpi_dz)
    dw_dt_ifc = np.zeros(len(w))
    dw_dt_ifc[1:-1] = 0.5 * (dw_dt_mc[:-1] + dw_dt_mc[1:])
    dw_dt_ifc[0] = dw_dt_mc[0]
    dw_dt_ifc[-1] = dw_dt_mc[-1]
    w_new = w + dt * dw_dt_ifc
    w_new[0] = 0.0
    w_new[-1] = 0.0

    drho = drho_dt(rho, w_new, inv_dz)
    rho_new = rho + dt * drho
    qv_advected = step_qv(rho, qv, w_new, inv_dz, dt)

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
    qr_new = qr + ac["qr"]   # добавляем прирост к уже существующему qr

    # 2. Осаждение (qr падает)
    qr_new, precip_flux = sedimentation_step(qr_new, Vt, dz, dt)

    # 3. Испарение (часть qr -> qv)
    ev = evaporation_step(qr_new, qv_new, qc_new, exner_new, dt, evap_coef=evap_coef)
    qr_new = ev["qr"]
    qv_new = ev["qv"]

    if use_diffusion:
        theta_v_diff = virtual_theta(theta_new, qv_new)
        Km = km_smagorinsky(theta_v_diff, w_new, z_mc, z_ifc, inv_dz, dz, dx=dx)
        theta_new = diffusion_step(theta_new, Km, z_mc, z_ifc, dz, dt)
        qv_new = diffusion_step(qv_new, Km, z_mc, z_ifc, dz, dt)
        qc_new = diffusion_step(qc_new, Km, z_mc, z_ifc, dz, dt)
        qv_new = np.maximum(qv_new, 0.0)
        qc_new = np.maximum(qc_new, 0.0)

    theta_v_final = virtual_theta(theta_new, qv_new)
    exner_final = diagnose_exner(rho_new, theta_v_final)

    return {
        "rho": rho_new,
        "w": w_new,
        "theta": theta_new,
        "qv": qv_new,
        "qc": qc_new,
        "qr": qr_new,
        "precip_flux": precip_flux,
        "theta_v": theta_v_final,
        "exner": exner_final,
        "pi_pr": perturbation(exner_final, exner_ref),
    }
