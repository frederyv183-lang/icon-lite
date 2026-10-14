"""
Диагностика тумана: приземное охлаждение, флаг тумана, видимость.

Радиационный туман:  выхолаживание поверхности за счёт излучения.
Адвективный туман:    тёплый влажный воздух над холодной поверхностью.

Туман = qc > 0 в приземном слое (z < z_threshold).
"""
import numpy as np

# Константы по умолчанию
K_RAD = 1.0e-4      # скорость радиационного охлаждения, 1/с
K_SURF = 5.0e-4     # коэффициент приземного теплообмена, 1/с
Z_FOG = 200.0       # порог приземного слоя для флага тумана, м
VIS_A = 0.02        # эмпирический коэффициент: vis [м] = VIS_A / qc [кг/кг]


def radiative_cooling_step(theta, dt, k_rad=K_RAD, theta_ref=None):
    """
    Радиационное выхолаживание приземного уровня (1D).

    dtheta[0]/dt = -k_rad * (theta[0] - theta_ref[0])

    Применяем только к первому уровню. Возвращает изменённый theta.
    """
    if theta_ref is None:
        theta_ref = theta
    theta_new = theta.copy()
    theta_new[0] = theta[0] - k_rad * (theta[0] - theta_ref[0]) * dt
    return theta_new


def surface_cooling_step_2d(theta, dt, U, k_surf=K_SURF, theta_surf=None):
    """
    Приземное охлаждение в 2D (адвективный туман).

    Тёплый воздух с температурой theta[0, :] движется со скоростью U (м/с)
    над поверхностью с температурой theta_surf. Скорость охлаждения
    пропорциональна U (быстрее принос → быстрее охлаждение).

    dtheta[0, :]/dt = -k_surf * U * (theta[0, :] - theta_surf)

    Возвращает изменённый theta (копию).
    """
    if theta_surf is None:
        theta_surf = theta[0, 0] - 5.0
    theta_new = theta.copy()
    theta_new[0, :] = theta[0, :] - k_surf * U * (theta[0, :] - theta_surf) * dt
    return theta_new


def diagnose_fog_flag(qc, z_mc, z_threshold=Z_FOG):
    """
    True, если есть облачная вода в приземном слое (z < z_threshold).

    qc : ndarray (nz,) или (nz, nx)
    z_mc : (nz,) — высоты
    Возвращает bool.
    """
    k_max = int(np.searchsorted(z_mc, z_threshold, side="right"))
    if k_max == 0:
        k_max = 1
    return bool(np.any(qc[:k_max] > 1e-6))


def visibility_from_qc(qc):
    """
    Эмпирическая видимость (м) по содержанию облачной воды qc (кг/кг).

    vis ≈ VIS_A / qc, где VIS_A = 0.02 — типичная формула для тёплого тумана.
    При qc → 0 видимость → бесконечность (заменяем на 20_000 м).
    """
    qc = np.asarray(qc, dtype=float)
    vis = np.where(qc > 1e-8, VIS_A / np.maximum(qc, 1e-8), 20_000.0)
    return np.minimum(vis, 20_000.0)


def fog_metrics(qc, z_mc):
    """
    Сводные метрики тумана.

    Возвращает:
      has_fog        — bool (есть ли qc в приземном слое)
      vis_min_m      — минимальная видимость в приземном слое (м)
      qc_surface_gkg — qc в первой ячейке (г/кг)
      top_fog_m      — верхняя граница тумана (м), где qc > 1e-6
    """
    qc = np.asarray(qc)
    k_max = int(np.searchsorted(z_mc, Z_FOG, side="right"))
    if k_max == 0:
        k_max = 1

    vis = visibility_from_qc(qc)
    vis_surf = vis[:k_max] if qc.ndim == 1 else vis[:k_max, :]

    has_fog = diagnose_fog_flag(qc, z_mc)

    qc_surface = float(qc[0]) if qc.ndim == 1 else float(qc[0].max())

    top_fog = 0.0
    if has_fog:
        if qc.ndim == 1:
            idx = np.where(qc > 1e-6)[0]
        else:
            idx = np.where(qc.max(axis=1) > 1e-6)[0]
        if len(idx):
            top_fog = float(z_mc[idx.max()])

    return {
        "has_fog": has_fog,
        "vis_min_m": float(vis_surf.min()) if vis_surf.size else 20_000.0,
        "qc_surface_gkg": qc_surface * 1000.0,
        "top_fog_m": top_fog,
    }