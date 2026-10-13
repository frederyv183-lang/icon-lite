"""
Вертикальная сетка и веса интерполяции.
Упрощённая версия из ICON (mo_vertical_grid.f90).
"""
import numpy as np


def make_vertical_grid(nlev, z_top=20000.0, stretch=1.15):
    """
    Создаёт вертикальную сетку с растяжением.

    Параметры
    ---------
    nlev : int
        Число уровней (центров ячеек).
    z_top : float
        Высота верхней границы модели, м.
    stretch : float
        Коэффициент растяжения шага с высотой (1.0 = равномерная).

    Возвращает
    ----------
    dict с полями:
        z_mc : (nlev,)      высоты центров ячеек
        z_ifc : (nlev+1,)   высоты интерфейсов
        dz : (nlev,)        толщины слоёв
        wgtfac_c : (nlev,)  веса интерполяции на интерфейсы
    """
    eta_ifc = np.linspace(0.0, 1.0, nlev + 1) ** stretch
    z_ifc = eta_ifc * z_top

    z_mc = 0.5 * (z_ifc[:-1] + z_ifc[1:])
    dz = np.diff(z_ifc)

    wgtfac_c = np.zeros(nlev)
    for k in range(1, nlev):
        wgtfac_c[k] = dz[k-1] / (dz[k-1] + dz[k])
    wgtfac_c[0] = 0.5
    wgtfac_c[-1] = 0.5

    return {
        "z_mc": z_mc,
        "z_ifc": z_ifc,
        "dz": dz,
        "wgtfac_c": wgtfac_c,
    }


def interp_to_ifc(field_mc, wgtfac_c):
    """
    Интерполяция поля с центров ячеек на интерфейсы.
    Аналог z_exner_ic в ICON.
    """
    field_mc = np.asarray(field_mc)
    nlev = field_mc.shape[0]

    out_shape = (nlev + 1,) + field_mc.shape[1:]
    field_ifc = np.zeros(out_shape, dtype=field_mc.dtype)

    for k in range(1, nlev):
        w = wgtfac_c[k]
        field_ifc[k] = w * field_mc[k] + (1.0 - w) * field_mc[k-1]

    field_ifc[0] = field_mc[0]
    field_ifc[-1] = field_mc[-1]

    return field_ifc
