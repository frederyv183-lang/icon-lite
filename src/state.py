"""
Контейнер прогностических переменных и опорное состояние.
Упрощённая версия из ICON (mo_nonhydro_state.f90, mo_nonhydro_types.f90).

Соглашения:
  - Вертикальный индекс k = 0..nlev-1 идёт СНИЗУ ВВЕРХ (k=0 — поверхность).
  - Горизонтальные размеры: ncell (центры), nedge (рёбра).

Уравнение состояния (форма ICON):
    p = rho * Rd * theta_v * pi^(c_vd/Rd)
    =>  rho = p0 * pi / (Rd * theta_v)
    =>  pi  = rho * Rd * theta_v / p0
"""
import numpy as np
from dataclasses import dataclass, field


# Физические константы
G = 9.80665          # ускорение свободного падения, м/с²
CPD = 1004.64        # удельная теплоёмкость сухого воздуха при пост. давлении
RD = 287.04          # газовая постоянная сухого воздуха
P0 = 100000.0        # опорное давление, Па


@dataclass
class State:
    """Все прогностические и диагностические поля модели."""
    rho: np.ndarray       # (nlev, ncell)  плотность
    vn: np.ndarray        # (nlev, nedge)  нормальная скорость
    w: np.ndarray         # (nlev+1, ncell) вертикальная скорость
    theta_v: np.ndarray   # (nlev, ncell)  виртуальная потенциальная температура
    exner: np.ndarray     # (nlev, ncell)  функция Экспера

    exner_ref: np.ndarray = field(default=None)
    rho_ref: np.ndarray = field(default=None)
    theta_ref: np.ndarray = field(default=None)

    exner_pr: np.ndarray = field(default=None)


def make_reference_atmosphere(z_mc, z_ifc, theta_surf=300.0, p_surf=101325.0,
                              z_trop=None, dtheta_dz_strat=2.0e-3):
    """
    Hydrostatically balanced atmosphere (troposphere + optional stratosphere).

    Parameters
    ----------
    z_mc, z_ifc : ndarray
        Heights of cell centers and interfaces.
    theta_surf : float
        Potential temperature at the surface, K.
    p_surf : float
        Pressure at the surface, Pa.
    z_trop : float or None
        Tropopause height, m. If None (default) -- no stratosphere,
        theta = const at all heights (legacy behavior).
        If set -- above z_trop, theta grows linearly with gradient
        dtheta_dz_strat.
    dtheta_dz_strat : float
        theta gradient above the tropopause, K/m. Default 2 K/km.

    Returns
    -------
    dict with:
        rho_ref   : (nlev,)
        theta_ref : (nlev,)
        exner_ref : (nlev,)
        z_trop_m  : float or None
        is_stable : bool
    """
    z_mc = np.asarray(z_mc, dtype=float)

    if z_trop is None:
        theta_ref = np.full_like(z_mc, theta_surf, dtype=float)
        exner_ref = np.exp(-G * z_mc / (CPD * theta_surf))
        z_trop_m = None
        is_stable = False
    else:
        theta_ref = np.where(z_mc < z_trop,
                             theta_surf,
                             theta_surf + dtheta_dz_strat * (z_mc - z_trop))
        ln_pi = np.empty_like(z_mc)
        ln_pi[0] = -G * z_mc[0] / (CPD * theta_ref[0])
        for k in range(1, len(z_mc)):
            dz = z_mc[k] - z_mc[k - 1]
            theta_mean = 0.5 * (theta_ref[k] + theta_ref[k - 1])
            ln_pi[k] = ln_pi[k - 1] - G * dz / (CPD * theta_mean)
        exner_ref = np.exp(ln_pi)
        z_trop_m = float(z_trop)
        is_stable = bool(dtheta_dz_strat > 0.0)

    rho_ref = P0 * exner_ref / (RD * theta_ref)

    return {
        "rho_ref": rho_ref,
        "theta_ref": theta_ref,
        "exner_ref": exner_ref,
        "z_trop_m": z_trop_m,
        "is_stable": is_stable,
    }

def diagnose_exner(rho, theta_v, p0=P0, rd=RD):
    """
    Диагностика Exner из rho·theta_v.
    Форма ICON: π = ρ·R_d·θ_v / p0
    """
    return rho * rd * theta_v / p0
