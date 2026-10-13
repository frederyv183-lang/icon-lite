"""
Уравнение неразрывности (1D).
Упрощённая версия из ICON (mo_solve_nonhydro.f90, блок 2129-2153).

Соглашение: k=0 — поверхность, k=nlev-1 — верх.
w живёт на интерфейсах (nlev+1 уровень), rho — в центрах (nlev уровней).
"""
import numpy as np


def mass_flux(rho, w):
    """
    Массовый поток rho*w на интерфейсах.

    Параметры
    ---------
    rho : (nlev, ...)   плотность в центрах ячеек
    w   : (nlev+1, ...) вертикальная скорость на интерфейсах

    Возвращает
    ----------
    mflx : (nlev+1, ...) массовый поток на интерфейсах
    """
    # интерполируем rho на интерфейсы (простое среднее)
    rho_ifc = np.zeros_like(w)
    rho_ifc[1:-1] = 0.5 * (rho[:-1] + rho[1:])
    rho_ifc[0] = rho[0]
    rho_ifc[-1] = rho[-1]
    return rho_ifc * w


def drho_dt(rho, w, inv_dz):
    """
    Тенденция плотности из уравнения неразрывности:
        d(rho)/dt = -d(rho*w)/dz

    В дискретной форме (снизу вверх, k=0 — поверхность):
        d(rho_k)/dt = -(mflx_{k+1} - mflx_k) / dz_k

    Параметры
    ---------
    rho : (nlev, ...)
    w : (nlev+1, ...)
    inv_dz : (nlev,) или (nlev, ...) = 1/Δz_k

    Возвращает
    ----------
    drho : (nlev, ...) тенденция плотности, кг/(м³·с)
    """
    mflx = mass_flux(rho, w)
    # разность потоков на верхней и нижней границе ячейки
    flux_div = mflx[1:] - mflx[:-1]  # (nlev, ...)

    if inv_dz.ndim == 1:
        inv_dz = inv_dz.reshape((inv_dz.shape[0],) + (1,) * (flux_div.ndim - 1))

    return -flux_div * inv_dz


def step_euler(rho, drho, dt):
    """rho^{n+1} = rho^n + dt · drho/dt."""
    return rho + dt * drho
