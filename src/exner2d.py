"""
2D-градиенты Exner: горизонтальный и вертикальный.
"""
import numpy as np
from grid2d import interp_to_edge


def perturbation_2d(exner, exner_ref_2d):
    """Возмущение Exner в 2D: pi' = pi - pi_ref."""
    return exner - exner_ref_2d


def exner_at_ifc_2d(exner_pr, wgtfac_c):
    """
    Exner на горизонтальных интерфейсах (вертикальная интерполяция).

    Формула:
        pi_ifc[k+1/2] = wgtfac_c[k+1] * pi_mc[k+1]
                      + (1 - wgtfac_c[k+1]) * pi_mc[k]
    для k = 0 .. nz-2 (внутренние интерфейсы)

    Параметры
    ---------
    exner_pr : (nz, nx)
    wgtfac_c : (nz,)

    Возвращает
    ----------
    exner_ifc : (nz+1, nx)
    """
    nz, nx = exner_pr.shape
    out = np.zeros((nz + 1, nx))

    # внутренние интерфейсы k=1..nz-1 (соответствуют границе между k-1 и k)
    # Для интерфейса с индексом k (1..nz-1):
    #   pi_ifc[k] = wgtfac_c[k] * pi_mc[k] + (1-wgtfac_c[k]) * pi_mc[k-1]
    w = wgtfac_c[1:, None]                # (nz-1, 1)
    out[1:-1, :] = w * exner_pr[1:, :] + (1.0 - w) * exner_pr[:-1, :]

    # края — экстраполяция копированием
    out[0, :] = exner_pr[0, :]
    out[-1, :] = exner_pr[-1, :]

    return out


def vertical_gradient_2d(exner_ifc, inv_dz):
    """
    Вертикальный градиент Exner в 2D: dpi'/dz.

    Формула:
        dpi_dz[k] = (pi_ifc[k+1] - pi_ifc[k]) / dz[k]

    Параметры
    ---------
    exner_ifc : (nz+1, nx)
    inv_dz : (nz,)

    Возвращает
    ----------
    dpi_dz : (nz, nx)
    """
    dpi_dz = exner_ifc[1:, :] - exner_ifc[:-1, :]
    return dpi_dz * inv_dz[:, None]


def horizontal_gradient_2d(exner_pr, dx_edge):
    """
    Горизонтальный градиент Exner на вертикальных гранях: dpi'/dx.

    Параметры
    ---------
    exner_pr : (nz, nx) — значения в центрах
    dx_edge : (nx+1,)   — расстояние между соседними центрами

    Возвращает
    ----------
    dpi_dx : (nz, nx+1) — значения на гранях
    """
    nz, nx = exner_pr.shape
    dpi_dx = np.zeros((nz, nx + 1))
    # внутренние грани: edge[i] между cell[i-1] и cell[i]
    dpi_dx[:, 1:-1] = (exner_pr[:, 1:] - exner_pr[:, :-1]) / dx_edge[1:-1][None, :]
    # на краях — односторонняя разность (копирование соседнего значения)
    dpi_dx[:, 0] = dpi_dx[:, 1]
    dpi_dx[:, -1] = dpi_dx[:, -2]
    return dpi_dx