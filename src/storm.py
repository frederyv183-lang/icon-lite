"""
розовая диагностика: CAPE, LI, светофор грозы.

CAPE (Convective Available Potential Energy) — запасённая энергия
плавучести для типичной восходящей частицы:
    CAPE = ∫_{LFC}^{EL} (g / θ_v_env) * (θ_v_part − θ_v_env) dz

LI (Lifted Index) — разность температур частицы и среды на 500 га:
    LI = T_env(500) − T_part(500)
    LI < −6 K — сильная неустойчивость, LI > 0 — устойчиво.

астица поднимается сухоадиабатически до LCL (уровень конденсации),
затем влажноадиабатически.
"""
import numpy as np

G = 9.80665          # ускорение свободного падения, м/с²
CPD = 1004.64        # теплоёмкость сухого воздуха, ж/(кг·)
RD = 287.04          # газовая постоянная сухого воздуха
EPS = 0.622          # R_d / R_v
LV = 2.5e6           # удельная теплота парообразования, ж/кг
P0 = 100000.0        # опорное давление, а
KAPPA = RD / CPD     # ≈ 0.2857
P_500 = 50000.0      # 500 га, а (стандартный уровень для LI)


def _esat_tetens(T_K):
    """авление насыщенного пара по Тетенсу, а."""
    return 610.78 * np.exp(17.27 * (T_K - 273.15) / (T_K - 35.85))


def _qsat(T_K, p_Pa):
    """Удельная влажность насыщения (кг/кг)."""
    es = _esat_tetens(T_K)
    return EPS * es / np.maximum(p_Pa - (1.0 - EPS) * es, 1.0)


def _pressure_from_z(z_m, z0=0.0, T0=288.15, Gamma=0.0065, p0=P0):
    """аротропная оценка давления по высоте (стандартная атмосфера).

    Fallback, если exner неизвестен.
    """
    T = T0 - Gamma * np.maximum(z_m - z0, 0.0)
    T = np.maximum(T, 200.0)
    return p0 * (T / T0) ** (G / (RD * Gamma))


def _pressure_from_exner(exner):
    """авление из Exner-функции (согласовано с моделью):
        p = p0 * exner^(cpd/Rd)
    """
    exner = np.asarray(exner, dtype=float)
    return P0 * np.maximum(exner, 1e-6) ** (1.0 / KAPPA)


def _theta_from_T_p(T_K, p_Pa):
    """отенциальная температура из T и p."""
    return T_K * (P0 / p_Pa) ** KAPPA


def _T_from_theta_p(theta, p_Pa):
    """Температура из потенциальной температуры и давления."""
    return theta * (p_Pa / P0) ** KAPPA


def _wet_adiabat_step(theta_part, qv_part, p_Pa, dp_Pa):
    """
    дин шаг влажноадиабатического подъёма.

    спользуем стандартный влажноадиабатический градиент:
        dT/dp = (1/p) * (Rd*T + Lv*r_s) / (cpd + Lv^2 * r_s * eps / (Rd*T^2))
    где r_s = q_sat/(1-q_sat) — массовое отношение насыщения.

    dp_Pa < 0 при подъёме, поэтому dT < 0 (охлаждение).
    """
    T_part = _T_from_theta_p(theta_part, p_Pa)
    qs = _qsat(T_part, p_Pa)
    rs = qs / max(1.0 - qs, 1e-8)

    num = RD * T_part + LV * rs
    den = CPD + (LV * LV * rs * EPS) / (RD * T_part * T_part)
    dT_dp = (1.0 / p_Pa) * num / den

    T_new = T_part + dT_dp * dp_Pa
    T_new = max(T_new, 150.0)
    qv_new = _qsat(T_new, p_Pa)
    theta_new = _theta_from_T_p(T_new, p_Pa)
    return theta_new, qv_new, T_new


def compute_cape(z_mc, theta_env, qv_env, exner_env,
                 theta_part_surf=None, qv_part_surf=None,
                 p_surf=None):
    """
    CAPE для типичной восходящей частицы.

    озвращает dict:
        cape_Jkg     : float            CAPE, ж/кг
        lfc_m        : float или None   уровень свободной конвекции, м
        el_m         : float или None   уровень равновесия, м
        z_lcl_m      : float            уровень конденсации, м
        cape_profile : (nlev,)          вклад в CAPE по слоям
    """
    z_mc = np.asarray(z_mc, dtype=float)
    theta_env = np.asarray(theta_env, dtype=float)
    qv_env = np.asarray(qv_env, dtype=float)
    exner_env = np.asarray(exner_env, dtype=float)
    nlev = len(z_mc)

    if theta_part_surf is None:
        theta_part_surf = float(theta_env[0])
    if qv_part_surf is None:
        qv_part_surf = float(qv_env[0])

    # авление из exner_env (согласовано с моделью)
    p_env = _pressure_from_exner(exner_env)
    theta_v_env = theta_env * (1.0 + 1.608 * qv_env) / (1.0 + qv_env)

    # --- тсечка CAPE по высоте ---
    # одель icon-lite не имеет стратосферы (theta_ref = const везде),
    # поэтому интегрируем CAPE только до z_top_cape = 14 км.
    Z_TOP_CAPE = 14000.0
    k_trop = int(np.searchsorted(z_mc, Z_TOP_CAPE, side="right")) - 1
    k_trop = max(1, min(k_trop, nlev - 1))
    z_trop = float(z_mc[k_trop])

    theta_v_part = np.empty(nlev)
    T_part_arr = np.empty(nlev)
    qv_part_arr = np.empty(nlev)
    theta_part_arr = np.empty(nlev)

    # ачальное состояние частицы
    T_p = _T_from_theta_p(theta_part_surf, p_env[0])
    qv_p = qv_part_surf
    saturated = False
    z_lcl = z_mc[0]

    # уровень 0
    if qv_p > _qsat(T_p, p_env[0]):
        saturated = True
        z_lcl = z_mc[0]
        qv_p = _qsat(T_p, p_env[0])
    theta_v_part[0] = T_p * (1.0 + 1.608 * qv_p) / (1.0 + qv_p)  # примерно
    T_part_arr[0] = T_p
    qv_part_arr[0] = qv_p
    theta_part_arr[0] = _theta_from_T_p(T_p, p_env[0])

    for k in range(1, nlev):
        p_k = p_env[k]
        dp = p_k - p_env[k - 1]  # < 0

        if not saturated:
            # Сухая адиабата: theta = const => T пересчитываем от theta
            theta_p = _theta_from_T_p(T_p, p_env[k - 1])
            T_k = _T_from_theta_p(theta_p, p_k)
            qs_k = _qsat(T_k, p_k)
            if qv_p > qs_k:
                # LCL между уровнями: переключаемся на влажную адиабату
                saturated = True
                z_lcl = z_mc[k]
                T_k, qv_k = _wet_step_T(T_p, qv_p, p_env[k - 1], p_k)
            else:
                qv_k = qv_p
        else:
            T_k, qv_k = _wet_step_T(T_p, qv_p, p_env[k - 1], p_k)

        T_p = T_k
        qv_p = qv_k
        T_part_arr[k] = T_k
        qv_part_arr[k] = qv_k
        theta_part_arr[k] = _theta_from_T_p(T_k, p_k)
        # иртуальная θ частицы — через T и qv
        th_vp = (T_k * (1.0 + 1.608 * qv_k) / (1.0 + qv_k)) * \
                (P0 / p_k) ** KAPPA
        # ыше тропопаузы CAPE не считаем (стратосфера стабильна)
        if k > k_trop:
            th_vp = theta_v_env[k] - 1.0  # заведомо < среды
        theta_v_part[k] = th_vp

    # --- CAPE = ∫ (g/θ_v_env) * (θ_v_part − θ_v_env) dz ---
    dz = np.empty(nlev)
    if nlev > 1:
        dz[0] = z_mc[1] - z_mc[0]
        dz[1:] = np.diff(z_mc)
    else:
        dz[0] = 1.0

    buoy = (G / np.maximum(theta_v_env, 1.0)) * (theta_v_part - theta_v_env)
    cape_layer = np.where(theta_v_part > theta_v_env, buoy * dz, 0.0)
    cape = float(np.sum(cape_layer))

    lfc_m = None
    el_m = None
    for k in range(nlev):
        if z_mc[k] >= z_lcl and theta_v_part[k] > theta_v_env[k]:
            lfc_m = float(z_mc[k])
            break
    if lfc_m is not None:
        for k in range(nlev - 1, -1, -1):
            if z_mc[k] > lfc_m and theta_v_part[k] > theta_v_env[k]:
                el_m = float(z_mc[k])
                break

    return {
        "cape_Jkg": cape,
        "lfc_m": lfc_m,
        "el_m": el_m,
        "z_lcl_m": float(z_lcl),
        "z_trop_m": z_trop,
        "cape_profile": cape_layer,
        "theta_v_part": theta_v_part,
        "theta_v_env": theta_v_env,
    }


def _wet_step_T(T_prev, qv_prev, p_prev, p_next):
    """
    дин шаг подъёма частицы с учётом обеих фаз.

    сли qv < qs(T, p) — сухая адиабата (theta = const), qv не меняется.
    сли qv >= qs(T, p) — влажная адиабата, избыток конденсируется,
    выделяется скрытое тепло, qv = qs(T, p).

    нтегрируем dT/dp по подшагам:
        dT/dp = (1/p) * (Rd*T + Lv*rs) / (cpd + Lv^2 * rs * eps / (Rd*T^2))
    где rs = qs/(1-qs).
    """
    T = T_prev
    qv = qv_prev
    p = p_prev
    n_sub = max(8, int(abs(p_next - p_prev) / 1000) + 1)
    dp = (p_next - p_prev) / n_sub

    for _ in range(n_sub):
        qs = _qsat(T, p)
        if qv <= qs:
            # сухая адиабата
            dTdp = KAPPA * T / p
            T_new = T + dTdp * dp
            qv_new = qv  # не меняется
        else:
            # влажная адиабата
            rs = qs / max(1.0 - qs, 1e-8)
            num = RD * T + LV * rs
            den = CPD + (LV * LV * rs * EPS) / (RD * T * T)
            dTdp = (1.0 / p) * num / den
            T_new = T + dTdp * dp
            qv_new = _qsat(T_new, p + dp)
            qv_new = min(qv, qv_new)
        T = max(T_new, 150.0)
        qv = qv_new
        p = p + dp

    return T, qv


def compute_li(z_mc, theta_env, qv_env, exner_env,
               theta_part_surf=None, qv_part_surf=None,
               p_surf=None, p_target=P_500):
    """
    Lifted Index на уровне p_target (по умолчанию 500 га).
    LI = T_env(p_target) − T_part(p_target)
    """
    z_mc = np.asarray(z_mc, dtype=float)
    theta_env = np.asarray(theta_env, dtype=float)
    qv_env = np.asarray(qv_env, dtype=float)
    nlev = len(z_mc)

    if theta_part_surf is None:
        theta_part_surf = float(theta_env[0])
    if qv_part_surf is None:
        qv_part_surf = float(qv_env[0])

    p_env = _pressure_from_exner(exner_env)
    T_env_K = _T_from_theta_p(theta_env, p_env)
    logp = np.log(p_env)
    idx = np.argsort(logp)
    T_env_target = float(np.interp(np.log(p_target), logp[idx], T_env_K[idx]))

    # одъём частицы до p_target
    T_p = _T_from_theta_p(theta_part_surf, p_env[0])
    qv_p = qv_part_surf
    saturated = False
    T_part_target = float("nan")

    for k in range(1, nlev):
        p_k = p_env[k]
        if p_k <= p_target:
            # достигли p_target между k-1 и k
            T_k, _ = _wet_step_T(T_p, qv_p, p_env[k - 1], p_target) \
                if saturated else (None, None)
            if not saturated:
                theta_p = _theta_from_T_p(T_p, p_env[k - 1])
                T_try = _T_from_theta_p(theta_p, p_target)
                if qv_p > _qsat(T_try, p_target):
                    T_k, _ = _wet_step_T(T_p, qv_p, p_env[k - 1], p_target)
                else:
                    T_k = T_try
            T_part_target = T_k
            break

        if not saturated:
            theta_p = _theta_from_T_p(T_p, p_env[k - 1])
            T_k = _T_from_theta_p(theta_p, p_k)
            if qv_p > _qsat(T_k, p_k):
                saturated = True
                T_k, qv_k = _wet_step_T(T_p, qv_p, p_env[k - 1], p_k)
                T_p, qv_p = T_k, qv_k
            else:
                T_p = T_k
        else:
            T_p, qv_p = _wet_step_T(T_p, qv_p, p_env[k - 1], p_k)

    return float(T_env_target - T_part_target)


def storm_index(cape_Jkg, li_K, max_w):
    """
    Светофор грозы.

    🟢 спокойно        CAPE < 300
    🟡 возможен ливень 300 ≤ CAPE < 1000
    🔴 гроза           CAPE ≥ 1000 и max|w| ≥ 15 м/с
    🟠 сильная гроза   CAPE ≥ 2500 и max|w| ≥ 25 м/с
    """
    if cape_Jkg >= 2500.0 and max_w >= 25.0:
        return {"level": "extreme", "color": "#d62828", "emoji": "🟠",
                "label": "Сильная гроза",
                "advice": "пасная конвекция. озможны шквалы, град, ливни."}
    if cape_Jkg >= 1000.0 and max_w >= 15.0:
        return {"level": "storm", "color": "#e63946", "emoji": "🔴",
                "label": "роза",
                "advice": "Условия для грозы. учево-дождевые облака, ливни."}
    if cape_Jkg >= 300.0:
        return {"level": "shower", "color": "#f4a261", "emoji": "🟡",
                "label": "озможен ливень",
                "advice": "Умеренная неустойчивость. озможны локальные ливни."}
    return {"level": "calm", "color": "#2a9d8f", "emoji": "🟢",
            "label": "Спокойно",
            "advice": "Устойчивая стратификация. роз не ожидается."}
