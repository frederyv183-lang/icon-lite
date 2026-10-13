"""
2D-состояние модели: прогностические переменные и опорная атмосфера.
Всё расширено на горизонтальную ось i.
"""
import numpy as np
from state import make_reference_atmosphere


def make_initial_2d(grid, theta_ref, qv_max=0.0, dtheta=2.0,
                    x0_km=20.0, z0_km=3.0, radius_x_km=5.0, radius_z_km=1.5):
    """
    Создаёт 2D-начальные условия: гауссов пузырь в плоскости (x, z).

    Параметры
    ---------
    grid : dict       выход make_grid2d
    theta_ref : (nz,) опорная theta
    qv_max : float    максимальная влажность, кг/кг
    dtheta : float    максимальное возмущение theta, К
    x0_km, z0_km : float   центр пузыря
    radius_x_km, radius_z_km : float  размеры пузыря

    Возвращает
    ----------
    dict с 2D-полями:
        rho : (nz, nx)
        w : (nz+1, nx)
        vn : (nz, nx+1)
        theta : (nz, nx)
        qv : (nz, nx)
        qc : (nz, nx)
    """
    nz = grid["nz"]
    nx = grid["nx"]
    z_mc = grid["z_mc"][:, None]      # (nz, 1)
    x_mc = grid["x_mc"][None, :]      # (1, nx)

    # опорная плотность 2D (одинаковая по x)
    ref = make_reference_atmosphere(z_mc.flatten(), grid["z_ifc"], 300.0)
    rho = np.tile(ref["rho_ref"][:, None], (1, nx))       # (nz, nx)
    theta_ref_2d = np.tile(ref["theta_ref"][:, None], (1, nx))

    # гауссов пузырь
    x0 = x0_km * 1000.0
    z0 = z0_km * 1000.0
    rx = radius_x_km * 1000.0
    rz = radius_z_km * 1000.0
    shape = np.exp(-(((x_mc - x0) / rx) ** 2 + ((z_mc - z0) / rz) ** 2))

    theta = theta_ref_2d + dtheta * shape
    qv = qv_max * shape
    qc = np.zeros_like(theta)

    # скорости — нули
    w = np.zeros((nz + 1, nx))
    vn = np.zeros((nz, nx + 1))

    return {
        "rho": rho,
        "w": w,
        "vn": vn,
        "theta": theta,
        "qv": qv,
        "qc": qc,
        "theta_ref_2d": theta_ref_2d,
    }