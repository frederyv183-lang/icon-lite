"""
2D-уравнение неразрывности.
    drho/dt = -d(rho*u)/dx - d(rho*w)/dz
"""
import numpy as np
from grid2d import interp_to_edge


def mass_flux_x(rho, vn):
    """
    Горизонтальный массовый поток rho*vn на вертикальных гранях.

    rho : (nz, nx)     плотность в центрах
    vn : (nz, nx+1)    нормальная скорость на гранях
    Возвращает : (nz, nx+1) поток rho*vn на гранях
    """
    rho_edge = interp_to_edge(rho)  # (nz, nx+1)
    return rho_edge * vn


def mass_flux_z(rho, w):
    """
    Вертикальный массовый поток rho*w на горизонтальных интерфейсах.

    rho : (nz, nx)
    w : (nz+1, nx)
    Возвращает : (nz+1, nx)
    """
    nz, nx = rho.shape
    rho_ifc = np.zeros((nz + 1, nx))
    rho_ifc[1:-1, :] = 0.5 * (rho[:-1, :] + rho[1:, :])
    rho_ifc[0, :] = rho[0, :]
    rho_ifc[-1, :] = rho[-1, :]
    return rho_ifc * w


def drho_dt_2d(rho, w, vn, dx, inv_dz):
    """
    2D-тенденция плотности:
        drho/dt = -(mflx_x[i+1] - mflx_x[i])/dx
                  -(mflx_z[k+1] - mflx_z[k])/dz

    Параметры
    ---------
    rho : (nz, nx)
    w : (nz+1, nx)
    vn : (nz, nx+1)
    dx : (nx,)        ширина ячейки
    inv_dz : (nz,)    1/dz

    Возвращает
    ----------
    drho : (nz, nx)
    """
    # потоки
    mflx_x = mass_flux_x(rho, vn)      # (nz, nx+1)
    mflx_z = mass_flux_z(rho, w)       # (nz+1, nx)

    # дивергенция по горизонтали
    div_x = (mflx_x[:, 1:] - mflx_x[:, :-1]) / dx[None, :]  # (nz, nx)

    # дивергенция по вертикали
    div_z = (mflx_z[1:, :] - mflx_z[:-1, :]) * inv_dz[:, None]  # (nz, nx)

    return -(div_x + div_z)