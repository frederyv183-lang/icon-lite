"""
Диагностика температуры воздуха.

T = theta * Pi,  где Pi = (p/p0)^(Rd/cpd) — Exner-функция.
"""
import numpy as np


def diagnose_temperature(theta, exner):
    """
    Температура воздуха T (К) из потенциальной температуры theta и Exner-функции Pi.

    Параметры
    ---------
    theta : ndarray     потенциальная температура, К
    exner : ndarray     Exner-функция (безразмерная), той же формы

    Возвращает
    ----------
    T : ndarray         температура, К
    """
    return theta * exner


def temperature_celsius(T_K):
    """Перевод Кельвинов в градусы Цельсия."""
    return T_K - 273.15


def extrapolate_t_2m(T_K, z_mc, z_target=2.0):
    """
    Линейная экстраполяция температуры к высоте z_target (по умолчанию 2 м).

    T_2m = T[0] + (T[1] - T[0]) * (z_target - z[0]) / (z[1] - z[0])

    Если z_target >= z[0], возвращаем T[0] (вырожденный случай).
    """
    T_K = np.asarray(T_K)
    z_mc = np.asarray(z_mc)
    if T_K.shape[0] < 2 or z_target >= z_mc[0]:
        return float(T_K[0])
    slope = (T_K[1] - T_K[0]) / (z_mc[1] - z_mc[0])
    return float(T_K[0] + slope * (z_target - z_mc[0]))


def extract_t_metrics(theta, exner, z_mc, qc=None):
    """
    Сводные метрики температуры для отображения.

    Возвращает словарь:
      T_surface_C  — температура в первой ячейке (центр), °C
      T_2m_C       — экстраполяция к 2 м, °C
      T_cloud_C    — температура на уровне максимума qc, °C (None если qc=0)
      T_min_C      — минимум по колонке, °C
      T_max_C      — максимум по колонке, °C
    """
    T_K = diagnose_temperature(theta, exner)
    T_C = temperature_celsius(T_K)

    T_surface_C = float(T_C[0])
    T_2m_C = temperature_celsius(extrapolate_t_2m(T_K, z_mc, z_target=2.0))

    T_cloud_C = None
    if qc is not None and np.max(qc) > 0.0:
        k_cloud = int(np.argmax(qc))
        T_cloud_C = float(T_C[k_cloud])

    return {
        "T_surface_C": T_surface_C,
        "T_2m_C": T_2m_C,
        "T_cloud_C": T_cloud_C,
        "T_min_C": float(np.min(T_C)),
        "T_max_C": float(np.max(T_C)),
    }