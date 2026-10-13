"""
Уравнение движения (вертикальная компонента).
Упрощённая версия из ICON (mo_solve_nonhydro.f90, блок 2180-2430).

ВАЖНО: работаем с возмущениями относительно опорного состояния.
Опорное состояние гидростатически сбалансировано.
Соглашение: k=0 — поверхность, k=nlev-1 — верх.
"""
import numpy as np
from state import CPD, G


def dw_dt_vertical(theta_v, theta_ref, dpi_dz):
    """
    Тенденция вертикальной скорости от возмущения барического градиента
    и плавучести.
    """
    pressure_term = -CPD * theta_v * dpi_dz
    buoyancy_term = G * (theta_v - theta_ref) / theta_ref
    return pressure_term + buoyancy_term


def step_euler(w, dw_dt, dt):
    """Шаг по времени методом Эйлера."""
    return w + dt * dw_dt
