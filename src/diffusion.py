"""
Турбулентная диффузия по схеме Смагоринского.
Упрощённая версия из ICON (mo_nh_diffusion.f90).
"""
import numpy as np
from state import G


CS = 0.15
RIC = 0.25
K_MIN = 1e-2
K_MAX = 1e3


def _reshape_for_broadcast(arr, target_ndim):
    """Добавляет оси справа, чтобы arr можно было broadcast'ить на target_ndim."""
    while arr.ndim < target_ndim:
        arr = arr[..., None]
    return arr


def deformation_vertical(w, inv_dz):
    """
    |S| = |dw/dz|.

    Параметры
    ---------
    w : (nz+1, ...)   вертикальная скорость на интерфейсах
    inv_dz : (nz,)    обратные толщины слоёв

    Возвращает
    ----------
    S : (nz, ...)
    """
    dw = w[1:] - w[:-1]                   # (nz, ...)
    inv_dz_r = _reshape_for_broadcast(inv_dz, dw.ndim)
    return np.abs(dw * inv_dz_r)


def richardson_number(theta_v, w, inv_dz, dz):
    """
    Локальное число Ричардсона:
        Ri = N^2 / |S|^2
    где N^2 = (g/theta) * dtheta/dz.

    Параметры
    ---------
    theta_v : (nz, ...)
    w : (nz+1, ...)
    inv_dz : (nz,)
    dz : (nz,)

    Возвращает
    ----------
    Ri : (nz, ...)
    """
    nz = theta_v.shape[0]

    # вертикальный градиент theta_v (центральная разность внутри, односторонняя по краям)
    dtheta_dz = np.zeros_like(theta_v)
    if nz > 2:
        dtheta_dz[1:-1] = (theta_v[2:] - theta_v[:-2]) / \
                          _reshape_for_broadcast(dz[2:] + dz[:-2], theta_v.ndim)
        dtheta_dz[0] = (theta_v[1] - theta_v[0]) / dz[0]
        dtheta_dz[-1] = (theta_v[-1] - theta_v[-2]) / dz[-1]

    N2 = (G / theta_v) * dtheta_dz           # квадрат частоты Брента-Вяйсяля
    S = deformation_vertical(w, inv_dz)      # (nz, ...)
    S2 = S ** 2 + 1e-12                      # защита от деления на 0

    Ri = N2 / S2
    return np.clip(Ri, -10.0, 10.0)


def km_smagorinsky(theta_v, w, z_mc, z_ifc, inv_dz, dz,
                   dx=500.0, cs=CS, ric=RIC):
    """
    Коэффициент турбулентности по Смагоринскому с подавлением по Ri.
    """
    # Delta = sqrt(dx * dz) — характерный размер ячейки
    Delta = np.sqrt(dx * dz)                 # (nz,)

    # деформация
    S = deformation_vertical(w, inv_dz)      # (nz, ...)

    # число Ричардсона
    Ri = richardson_number(theta_v, w, inv_dz, dz)

    # фактор устойчивости
    F = np.maximum(0.0, 1.0 - Ri / ric)

    # коэффициент
    Delta_r = _reshape_for_broadcast(Delta, theta_v.ndim)
    Km = (cs * Delta_r) ** 2 * S * F

    return np.clip(Km, K_MIN, K_MAX)

def diffusion_step(field, Km, z_mc, z_ifc, dz, dt):
    """
    Один шаг диффузии для поля field с коэффициентом Km.

    Формула:
        dfield/dt = d/dz ( Km * dfield/dz )

    Дискретно:
        F_{k+1/2} = Km_{k+1/2} * (field_{k+1} - field_k) / dz_{k+1/2}
        dfield_k/dt = (F_{k+1/2} - F_{k-1/2}) / dz_k

    Параметры
    ---------
    field : (nz, ...)    поле в центрах ячеек
    Km : (nz, ...)       коэффициент турбулентности в центрах
    z_mc : (nz,)         высоты центров
    z_ifc : (nz+1,)      высоты интерфейсов
    dz : (nz,)           толщины слоёв
    dt : float           шаг по времени

    Возвращает
    ----------
    field_new : (nz, ...)  обновлённое поле
    """
    nz = field.shape[0]

    # интерполяция Km на интерфейсы
    Km_ifc = np.zeros((nz + 1,) + field.shape[1:])
    if nz > 1:
        Km_ifc[1:-1] = 0.5 * (Km[:-1] + Km[1:])
        Km_ifc[0] = Km[0]
        Km_ifc[-1] = Km[-1]

    # толщина слоя между центрами (для градиента)
    dz_ifc = np.zeros(nz + 1)
    dz_ifc[1:-1] = 0.5 * (dz[:-1] + dz[1:])
    dz_ifc[0] = dz[0]
    dz_ifc[-1] = dz[-1]
    dz_ifc_r = dz_ifc.reshape((nz + 1,) + (1,) * (field.ndim - 1))

    # градиент поля на интерфейсах
    dfield_dz_ifc = np.zeros((nz + 1,) + field.shape[1:])
    dfield_dz_ifc[1:-1] = (field[1:] - field[:-1]) / dz_ifc_r[1:-1]

    # поток на интерфейсах
    flux_ifc = Km_ifc * dfield_dz_ifc

    # дивергенция потока
    flux_div = np.zeros_like(field)
    flux_div = (flux_ifc[1:] - flux_ifc[:-1]) / dz.reshape(
        (nz,) + (1,) * (field.ndim - 1)
    )

    return field + dt * flux_div
