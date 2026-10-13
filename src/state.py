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


def make_reference_atmosphere(z_mc, z_ifc, theta_surf=300.0, p_surf=101325.0):
    """
    Гидростатически равновесная атмосфера с θ = const.

    Возвращает:
        rho_ref   : (nlev,)
        theta_ref : (nlev,)
        exner_ref : (nlev,)

    Формулы:
        θ(z)   = θ_surf
        π(z)   = exp(-g·z / (c_pd · θ_surf))
        ρ(z)   = p0 · π(z) / (R_d · θ_surf)      <-- форма ICON
    """
    theta_ref = np.full_like(z_mc, theta_surf, dtype=float)

    # Exner из гидростатики: dπ/dz = -g / (c_pd · θ)
    exner_ref = np.exp(-G * z_mc / (CPD * theta_surf))

    # плотность из уравнения состояния в форме ICON
    rho_ref = P0 * exner_ref / (RD * theta_ref)

    return {
        "rho_ref": rho_ref,
        "theta_ref": theta_ref,
        "exner_ref": exner_ref,
    }


def diagnose_exner(rho, theta_v, p0=P0, rd=RD):
    """
    Диагностика Exner из rho·theta_v.
    Форма ICON: π = ρ·R_d·θ_v / p0
    """
    return rho * rd * theta_v / p0
