"""
Двумерная сетка (x, z).
Расширение одномерной grid.py на горизонтальное направление.

Соглашения:
  - Индексы: k (вертикаль, 0=поверхность), i (горизонталь, 0=левый край).
  - Скалярные поля (rho, theta, ...) живут в ЦЕНТРАХ ячеек: (nlev, ncell).
  - Нормальная скорость vn живёт на ВЕРТИКАЛЬНЫХ ГРАНЯХ: (nlev, nedge).
  - Вертикальная скорость w живёт на ГОРИЗОНТАЛЬНЫХ ИНТЕРФЕЙСАХ: (nlev+1, ncell).
"""
import numpy as np
from grid import make_vertical_grid


def make_grid2d(nz, nx, x_total=40000.0, z_top=20000.0, stretch=1.15):
    """
    Двумерная сетка (x, z).

    Параметры
    ---------
    nz : int
        число вертикальных уровней (центров ячеек)
    nx : int
        число горизонтальных ячеек
    x_total : float
        полная ширина области, м (по умолчанию 40 км)
    z_top : float
        высота верхней границы, м
    stretch : float
        коэффициент растяжения по вертикали

    Возвращает
    ----------
    dict с:
        z_mc : (nz,)       высоты центров ячеек
        z_ifc : (nz+1,)    высоты интерфейсов
        dz : (nz,)         толщины слоёв
        wgtfac_c : (nz,)   веса интерполяции центр→интерфейс

        x_mc : (nx,)       координаты центров ячеек
        x_edge : (nx+1,)   координаты граней
        dx : (nx,)         ширины ячеек
        dx_edge : (nx+1,)  расстояния между центрами соседних ячеек

        nz, nx : int
    """
    # вертикальная сетка из существующего модуля
    vgrid = make_vertical_grid(nlev=nz, z_top=z_top, stretch=stretch)

    # горизонтальная сетка (равномерная)
    x_edge = np.linspace(0.0, x_total, nx + 1)
    dx = np.diff(x_edge)
    x_mc = 0.5 * (x_edge[:-1] + x_edge[1:])

    # расстояние между центрами соседних ячеек (для градиентов на грани)
    dx_edge = np.zeros(nx + 1)
    dx_edge[1:-1] = x_mc[1:] - x_mc[:-1]  # внутри
    dx_edge[0] = dx_edge[1]              # на левой границе — копия
    dx_edge[-1] = dx_edge[-2]            # на правой — копия

    return {
        "z_mc": vgrid["z_mc"],
        "z_ifc": vgrid["z_ifc"],
        "dz": vgrid["dz"],
        "wgtfac_c": vgrid["wgtfac_c"],
        "x_mc": x_mc,
        "x_edge": x_edge,
        "dx": dx,
        "dx_edge": dx_edge,
        "nz": nz,
        "nx": nx,
    }


def interp_to_edge(field_mc):
    """
    Интерполяция скалярного поля из центров ячеек на вертикальные грани (рёбра).
    Простое среднее.

    Параметры
    ---------
    field_mc : (nz, nx)   значения в центрах
        Или (nz, nx, ...) для многокомпонентных полей

    Возвращает
    ----------
    field_edge : (nz, nx+1)
        Значения на гранях. На краях — копия крайних центров.
    """
    field_mc = np.asarray(field_mc)
    nz, nx = field_mc.shape[:2]
    out_shape = (nz, nx + 1) + field_mc.shape[2:]
    field_edge = np.zeros(out_shape, dtype=field_mc.dtype)

    field_edge[:, 1:-1] = 0.5 * (field_mc[:, :-1] + field_mc[:, 1:])
    field_edge[:, 0] = field_mc[:, 0]
    field_edge[:, -1] = field_mc[:, -1]

    return field_edge


def interp_to_center_from_edge(field_edge):
    """
    Интерполяция с рёбер на центры ячеек.
    Каждая ячейка получает среднее своих двух граней.

    Параметры
    ---------
    field_edge : (nz, nx+1)

    Возвращает
    ----------
    field_mc : (nz, nx)
    """
    field_edge = np.asarray(field_edge)
    return 0.5 * (field_edge[:, :-1] + field_edge[:, 1:])