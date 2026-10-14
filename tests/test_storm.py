"""нит-тесты для src/storm.py."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from storm import (compute_cape, compute_li, storm_index,
                   _qsat, _esat_tetens, _pressure_from_exner,
                   _wet_step_T, _theta_from_T_p, _T_from_theta_p,
                   P0, KAPPA, P_500)


# ---------- вспомогательное ----------

def _make_atmosphere(nlev=40, z_top=20000.0, stretch=1.15,
                     theta_surf=300.0):
    """Стандартная атмосфера icon-lite + опциональный нагрев."""
    from grid import make_vertical_grid
    from state import make_reference_atmosphere
    g = make_vertical_grid(nlev=nlev, z_top=z_top, stretch=stretch)
    ref = make_reference_atmosphere(g["z_mc"], g["z_ifc"],
                                    theta_surf=theta_surf)
    return g, ref


# ---------- 1. Формулы ----------

def test_esat_tetens_at_273():
    """e_sat(273.15 K) ≈ 610.78 а (по определению)."""
    es = _esat_tetens(273.15)
    assert abs(es - 610.78) < 0.5


def test_qsat_at_300K_1000hPa():
    """q_sat(300 K, 1000 га) ≈ 22 г/кг (типичное значение)."""
    qs = _qsat(300.0, 100000.0)
    assert 0.020 < qs < 0.024


def test_pressure_from_exner_roundtrip():
    """p = P0 · exner^(1/KAPPA) — обратное к exner = (p/P0)^KAPPA."""
    p = 85000.0
    exner = (p / P0) ** KAPPA
    p_back = _pressure_from_exner(exner)
    assert abs(p_back - p) < 1.0


# ---------- 2. одъём частицы ----------

def test_wet_step_dry_then_saturated():
    """ри ненасыщенной частице qv не меняется; при насыщенной — падает."""
    # ненасыщенная: qv = 0.005, при T=280 K q_sat ≈ 0.010 > qv
    T_dry, qv_dry = _wet_step_T(280.0, 0.005, 90000.0, 85000.0)
    assert qv_dry == 0.005  # сухая адиабата — влага не тратится
    # насыщенная: qv = 0.020 при T=300 K, q_sat ≈ 0.022 — почти насыщена
    T_sat, qv_sat = _wet_step_T(300.0, 0.030, 100000.0, 95000.0)
    assert qv_sat < 0.030  # часть сконденсировалась
    assert T_sat < 300.0   # охлаждение


def test_wet_step_monotonic():
    """ем больше перепад давления, тем сильнее охлаждение."""
    T1, _ = _wet_step_T(300.0, 0.015, 100000.0, 95000.0)
    T2, _ = _wet_step_T(300.0, 0.015, 100000.0, 90000.0)
    assert T2 < T1  # больший подъём — холоднее


# ---------- 3. CAPE ----------

def test_cape_zero_for_dry_neutral_atmosphere():
    """Сухая нейтральная атмосфера (qv = 0, theta = theta_ref) → CAPE ≈ 0.

    римечание: theta_ref = const в icon-lite даёт нейтральную стратификацию
    по сухой адиабате; с ненулевой влажностью частица становится условно
    неустойчивой. оэтому для проверки "нет неустойчивости" берём qv = 0.
    """
    g, ref = _make_atmosphere()
    theta = ref["theta_ref"].copy()
    qv = np.zeros_like(theta)
    res = compute_cape(g["z_mc"], theta, qv, ref["exner_ref"],
                       theta_part_surf=float(theta[0]),
                       qv_part_surf=0.0)
    assert res["cape_Jkg"] < 10.0


def test_cape_positive_for_unstable():
    """Тёплая влажная частица → CAPE >> 0."""
    g, ref = _make_atmosphere()
    theta_env = ref["theta_ref"].copy()
    qv_env = 0.005 * np.exp(-g["z_mc"] / 2500.0)  # сухая среда
    theta_part = float(theta_env[0]) + 4.0
    qv_part = 0.018  # влажная частица
    res = compute_cape(g["z_mc"], theta_env, qv_env, ref["exner_ref"],
                       theta_part_surf=theta_part,
                       qv_part_surf=qv_part)
    assert res["cape_Jkg"] > 500.0
    assert res["lfc_m"] is not None
    assert res["el_m"] is not None
    assert res["lfc_m"] < res["el_m"]


def test_cape_profile_sums_to_cape():
    """cape_profile.sum() * dz ≈ CAPE (с точностью численного интегрирования)."""
    g, ref = _make_atmosphere()
    theta_env = ref["theta_ref"].copy()
    qv_env = 0.005 * np.exp(-g["z_mc"] / 2500.0)
    theta_part = float(theta_env[0]) + 3.0
    qv_part = 0.015
    res = compute_cape(g["z_mc"], theta_env, qv_env, ref["exner_ref"],
                       theta_part_surf=theta_part,
                       qv_part_surf=qv_part)
    # cape_profile уже взвешен на dz внутри compute_cape
    assert abs(float(np.sum(res["cape_profile"])) - res["cape_Jkg"]) < 1.0


def test_cape_bounded_by_z_top_cape():
    """EL не выше 14 км (отсечка модели)."""
    g, ref = _make_atmosphere()
    theta_env = ref["theta_ref"].copy()
    qv_env = 0.005 * np.exp(-g["z_mc"] / 2500.0)
    theta_part = float(theta_env[0]) + 5.0
    qv_part = 0.020
    res = compute_cape(g["z_mc"], theta_env, qv_env, ref["exner_ref"],
                       theta_part_surf=theta_part,
                       qv_part_surf=qv_part)
    if res["el_m"] is not None:
        assert res["el_m"] <= 14500.0


# ---------- 4. LI ----------

def test_li_zero_for_dry_neutral_atmosphere():
    """Сухая нейтральная атмосфера (qv = 0, theta = const) → LI ≈ 0.

    астица и среда одинаковы, влаги нет — на 500 га T_part = T_env.
    """
    g, ref = _make_atmosphere()
    theta = ref["theta_ref"].copy()
    qv = np.zeros_like(theta)
    li = compute_li(g["z_mc"], theta, qv, ref["exner_ref"],
                    theta_part_surf=float(theta[0]),
                    qv_part_surf=0.0)
    assert abs(li) < 2.0


def test_li_negative_for_unstable():
    """Тёплая влажная частица → LI < 0."""
    g, ref = _make_atmosphere()
    theta_env = ref["theta_ref"].copy()
    qv_env = 0.005 * np.exp(-g["z_mc"] / 2500.0)
    theta_part = float(theta_env[0]) + 4.0
    qv_part = 0.018
    li = compute_li(g["z_mc"], theta_env, qv_env, ref["exner_ref"],
                    theta_part_surf=theta_part,
                    qv_part_surf=qv_part)
    assert li < 0.0


# ---------- 5. Светофор ----------

def test_storm_index_levels():
    """роверка всех четырёх уровней светофора."""
    assert storm_index(100.0, 0.0, 5.0)["level"] == "calm"
    assert storm_index(500.0, -2.0, 8.0)["level"] == "shower"
    assert storm_index(1500.0, -5.0, 18.0)["level"] == "storm"
    assert storm_index(3000.0, -9.0, 30.0)["level"] == "extreme"


def test_storm_index_needs_both_cape_and_w():
    """ольшой CAPE, но слабый w → не гроза."""
    # CAPE = 1500, но w = 5 м/с — не хватает для "storm"
    assert storm_index(1500.0, -5.0, 5.0)["level"] == "shower"
    # CAPE = 800, w = 20 — всё равно только "shower"
    assert storm_index(800.0, -3.0, 20.0)["level"] == "shower"
