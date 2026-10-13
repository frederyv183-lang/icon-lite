"""
Градиенты Exner pressure.
Упрощённая версия из ICON (mo_solve_nonhydro.f90, строки 565-580, 1121-1240).

Обозначения:
    pi   = exner (полное)
    pi'  = exner - exner_ref  (возмущение)
    pi_ifc = pi' на интерфейсах
    dpi_dz = вертикальный градиент pi'
"""
import numpy as np
from grid import interp_to_ifc


def perturbation(exner, exner_ref):
    """
    Возмущение Exner: pi' = pi - pi_ref.

    Параметры
    ---------
    exner : (nlev, ncell) или (nlev,)
    exner_ref : (nlev,) или скаляр

    Возвращает
    ----------
    pi_pr : той же формы, что exner
    """
    return exner - exner_ref


def exner_at_ifc(exner_pr, wgtfac_c):
    """
    Exner на интерфейсах (полуровнях).
    Аналог z_exner_ic из ICON.

    Параметры
    ---------
    exner_pr : (nlev, ncell) или (nlev,)
    wgtfac_c : (nlev,)

    Возвращает
    ----------
    exner_ifc : (nlev+1, ...) 
    """
    return interp_to_ifc(exner_pr, wgtfac_c)


def vertical_gradient(exner_ifc, inv_dz):
    """
    Вертикальный градиент Exner: dpi'/dz.

    Формула ICON (строка 570-580):
        dpi_dz[k] = (pi_ifc[k] - pi_ifc[k+1]) * inv_dz[k]

    Здесь k=0..nlev-1 (снизу вверх), pi_ifc имеет nlev+1 уровней.

    Параметры
    ---------
    exner_ifc : (nlev+1, ...) значения pi' на интерфейсах
    inv_dz : (nlev,) или (nlev, ...) обратная толщина слоя 1/Δz_k

    Возвращает
    ----------
    dpi_dz : (nlev, ...) вертикальный градиент в центрах ячеек
    """
    exner_ifc = np.asarray(exner_ifc)
    nlev = exner_ifc.shape[0] - 1

    # dpi_dz[k] = (pi_ifc[k+1] - pi_ifc[k]) / dz[k]
    # В соглашении "снизу вверх": интерфейс k — нижняя граница ячейки k
    dpi_dz = (exner_ifc[1:] - exner_ifc[:-1])
    # умножаем на inv_dz, поддерживая broadcast
    if inv_dz.ndim == 1:
        # inv_dz: (nlev,)
        inv_dz = inv_dz.reshape((nlev,) + (1,) * (dpi_dz.ndim - 1))
    dpi_dz = dpi_dz * inv_dz

    return dpi_dz
