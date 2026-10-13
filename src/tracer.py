"""
Перенос трассеров (водяной пар и др.) вместе с массой.
Упрощённая версия из ICON (блок подготовки к tracer advection,
mo_solve_nonhydro.f90, строки 2513-2550).

Ключевая идея: используем ТОТ ЖЕ массовый поток, что и для rho.
Это гарантирует consistency with continuity (CWC) — согласованность с
уравнением неразрывности.
"""
import numpy as np
from continuity import mass_flux


def dqv_dt(rho, qv, w, inv_dz):
    """
    Тенденция rho*qv из уравнения сохранения массы воды:
        d(rho*qv)/dt = -d(rho*qv*w)/dz

    Затем qv_new = (rho*qv)_new / rho_new — обновление удельной влажности.

    Параметры
    ---------
    rho : (nlev, ...)     плотность (влажного воздуха)
    qv : (nlev, ...)      удельная влажность
    w : (nlev+1, ...)     вертикальная скорость на интерфейсах
    inv_dz : (nlev,)      1/dz

    Возвращает
    ----------
    dqv_dt : (nlev, ...)  тенденция qv, кг/(кг·с)
    """
    # масса воды в ячейке: rho * qv
    rho_qv = rho * qv

    # тот же поток массы, что и в continuity, но пропущенный через qv.
    # Для этого интерполируем qv на интерфейсы, умножаем на массовый поток
    # (который уже есть в mass_flux) и берём дивергенцию.

    # Поток массы (уже с интерполяцией rho на интерфейсы)
    mflx = mass_flux(rho, w)  # (nlev+1, ...)

    # qv на интерфейсах (простое среднее)
    qv_ifc = np.zeros_like(mflx)
    qv_ifc[1:-1] = 0.5 * (qv[:-1] + qv[1:])
    qv_ifc[0] = qv[0]
    qv_ifc[-1] = qv[-1]

    # поток воды на интерфейсах
    wflx = mflx * qv_ifc

    # дивергенция потока воды по ячейке
    flux_div = wflx[1:] - wflx[:-1]  # (nlev, ...)

    if inv_dz.ndim == 1:
        inv_dz = inv_dz.reshape((inv_dz.shape[0],) + (1,) * (flux_div.ndim - 1))

    # тенденция удельной влажности: d(rho*qv)/dt / rho
    drho_qv = -flux_div * inv_dz

    # чтобы получить dqv/dt, делим на rho (используем текущую rho как приближение)
    return drho_qv / rho


def step_qv(rho, qv, w, inv_dz, dt):
    """
    Один шаг по времени для qv.

    Обновление:
        rho_qv_new = rho_qv_old + dt * d(rho_qv)/dt
        qv_new = rho_qv_new / rho
    """
    rho_qv = rho * qv
    drho_qv = -_flux_divergence(rho, qv, w, inv_dz)
    rho_qv_new = rho_qv + dt * drho_qv
    # сохраняем неотрицательность
    rho_qv_new = np.maximum(rho_qv_new, 0.0)
    return rho_qv_new / rho


def _flux_divergence(rho, qv, w, inv_dz):
    """Внутренняя функция: дивергенция потока rho*qv*w."""
    mflx = mass_flux(rho, w)
    qv_ifc = np.zeros_like(mflx)
    qv_ifc[1:-1] = 0.5 * (qv[:-1] + qv[1:])
    qv_ifc[0] = qv[0]
    qv_ifc[-1] = qv[-1]
    wflx = mflx * qv_ifc
    flux_div = wflx[1:] - wflx[:-1]

    if inv_dz.ndim == 1:
        inv_dz = inv_dz.reshape((inv_dz.shape[0],) + (1,) * (flux_div.ndim - 1))

    return flux_div * inv_dz