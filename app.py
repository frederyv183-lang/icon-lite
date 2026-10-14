"""
Веб-приложение icon-lite: 1D- и 2D-модель атмосферы.
"""
import sys
import os
import io
import base64
sys.path.insert(0, "src")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, render_template, render_template_string, request, Response
import csv

from grid import make_vertical_grid
from grid2d import make_grid2d
from state import make_reference_atmosphere, diagnose_exner
from state2d import make_initial_2d
from timestep import step_forward
from timestep2d import step_forward_2d
from thermo import virtual_theta
from plotting2d import make_snapshots_2d
from plotting_t import make_snapshots_temperature
from diagnostics import (diagnose_temperature, temperature_celsius,
                         extrapolate_t_2m, extract_t_metrics)
from fog import (surface_cooling_step_2d, diagnose_fog_flag,
                 visibility_from_qc, fog_metrics)
from storm import compute_cape, compute_li, storm_index
from plotting_storm import (make_storm_profile,
                           make_storm_profile_bytes,
                           make_storm_timeseries)
from storm2d import compute_cape_2d, storm2d_metrics
from plotting_storm2d import make_storm2d_map, make_storm2d_profile

app = Flask(__name__)


def _is_dark():
    """True, если активна футуристическая (тёмная) тема."""
    return request.cookies.get("icon_lite_theme") == "futuristic"


HTML_1D = '''
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>icon-lite 1D</title>
<style>
  body { font-family: sans-serif; max-width: 1200px; margin: 2em auto;
         background: #f5f5f7; color: #222; }
  h1 { font-weight: 300; }
  nav { margin-bottom: 1.5em; }
  nav a { margin-right: 1em; color: #2b6cb0; text-decoration: none; }
  nav a:hover { text-decoration: underline; }
  .panel { background: white; padding: 1.5em; border-radius: 10px;
           box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 1.5em; }
  label { display: inline-block; margin-right: 1em; }
  input[type=number] { width: 80px; padding: 4px; }
  input[type=checkbox] { transform: scale(1.2); margin-left: 0.5em; }
  button { background: #2b6cb0; color: white; border: none;
           padding: 0.7em 1.5em; border-radius: 6px; cursor: pointer;
           font-size: 1em; }
  button:hover { background: #2c5282; }
  img { max-width: 100%; border-radius: 8px; }
  .meta { color: #555; font-size: 0.9em; }
</style>
</head>
<body>
<h1>icon-lite — 1D-модель</h1>
<nav><a href="/">1D вертикальная колонка</a><a href="/2d">2D конвекция</a><a href="/compare">Сравнение 1D/2D</a></nav>
<div class="panel">
  <p>Модель вертикальной колонки атмосферы с влажностью, конденсацией и турбулентностью.</p>
  <form method="post">
    <label>Уровни: <input type="number" name="nlev" value="{{ nlev }}" min="10" max="100"></label>
    <label>dθ, K: <input type="number" name="dtheta" value="{{ dtheta }}" step="0.5" min="0" max="20"></label>
    <label>qv, г/кг: <input type="number" name="qv_max" value="{{ qv_max }}" step="1" min="0" max="30"></label>
    <label>z пузыря, км: <input type="number" name="zbubble" value="{{ zbubble }}" step="0.5" min="1" max="15"></label>
    <label>dt, с: <input type="number" name="dt" value="{{ dt }}" step="0.05" min="0.05" max="2"></label>
    <label>Шагов: <input type="number" name="nsteps" value="{{ nsteps }}" min="100" max="5000"></label>
    <label>Турбулентность: <input type="checkbox" name="use_diffusion" {% if use_diffusion %}checked{% endif %}></label>
    <br><br>
    <button type="submit">Запустить 1D</button>
  </form>
</div>
{% if image %}
<div class="panel">
  <h2>Эволюция конвективного облака (1D)</h2>
  <p class="meta">max|w| = {{ max_w }} м/с | max qv = {{ max_qv }} г/кг |
     max qc = {{ max_qc }} г/кг | max qr = {{ max_qr }} г/кг |
     max dθ = {{ max_dtheta }} K</p>
  <img src="data:image/png;base64,{{ image }}" alt="1D">
  <p><a href="/download/1d?nlev={{ nlev }}&dtheta={{ dtheta }}&qv_max={{ qv_max }}&zbubble={{ zbubble }}&dt={{ dt }}&nsteps={{ nsteps }}&use_diffusion={{ 1 if use_diffusion else 0 }}">
     Скачать CSV</a></p>
</div>
{% endif %}
</body>
</html>
'''

HTML_2D = '''
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>icon-lite 2D</title>
<style>
  body { font-family: sans-serif; max-width: 1400px; margin: 2em auto;
         background: #f5f5f7; color: #222; }
  h1 { font-weight: 300; }
  nav { margin-bottom: 1.5em; }
  nav a { margin-right: 1em; color: #2b6cb0; text-decoration: none; }
  nav a:hover { text-decoration: underline; }
  .panel { background: white; padding: 1.5em; border-radius: 10px;
           box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 1.5em; }
  label { display: inline-block; margin-right: 1em; }
  input[type=number] { width: 80px; padding: 4px; }
  button { background: #2b6cb0; color: white; border: none;
           padding: 0.7em 1.5em; border-radius: 6px; cursor: pointer;
           font-size: 1em; }
  button:hover { background: #2c5282; }
  img { max-width: 100%; border-radius: 8px; }
  .meta { color: #555; font-size: 0.9em; }
</style>
</head>
<body>
<h1>icon-lite — 2D-модель конвекции</h1>
<nav><a href="/">1D вертикальная колонка</a><a href="/2d">2D конвекция</a><a href="/compare">Сравнение 1D/2D</a></nav>
<div class="panel">
  <p>Двумерная модель в плоскости (x, z).</p>
  <form method="post">
    <label>Вертикальные уровни: <input type="number" name="nz" value="{{ nz }}" min="10" max="100"></label>
    <label>Горизонтальные ячейки: <input type="number" name="nx" value="{{ nx }}" min="20" max="200"></label>
    <label>Ширина области, км: <input type="number" name="x_total" value="{{ x_total }}" step="5" min="10" max="200"></label>
    <label>dθ, K: <input type="number" name="dtheta" value="{{ dtheta }}" step="0.5" min="0" max="20"></label>
    <label>qv, г/кг: <input type="number" name="qv_max" value="{{ qv_max }}" step="1" min="0" max="30"></label>
    <label>z пузыря, км: <input type="number" name="zbubble" value="{{ zbubble }}" step="0.5" min="1" max="15"></label>
    <label>радиус x, км: <input type="number" name="radius_x" value="{{ radius_x }}" step="0.5" min="1" max="20"></label>
    <label>dt, с: <input type="number" name="dt" value="{{ dt }}" step="0.05" min="0.05" max="2"></label>
    <label>Шагов: <input type="number" name="nsteps" value="{{ nsteps }}" min="100" max="2000"></label>
    <br><br>
    <button type="submit">Запустить 2D</button>
  </form>
</div>
{% if image %}
<div class="panel">
  <h2>Эволюция конвективного облака (2D)</h2>
  <p class="meta">max|w| = {{ max_w }} м/с | max|vn| = {{ max_vn }} м/с |
     max qc = {{ max_qc }} г/кг | max qr = {{ max_qr }} г/кг</p>
  <img src="data:image/png;base64,{{ image }}" alt="2D">
</div>
{% endif %}
</body>
</html>
'''


HTML_COMPARE = '''
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>icon-lite — сравнение 1D и 2D</title>
<style>
  body { font-family: sans-serif; max-width: 1400px; margin: 2em auto;
         background: #f5f5f7; color: #222; }
  h1 { font-weight: 300; }
  nav { margin-bottom: 1.5em; }
  nav a { margin-right: 1em; color: #2b6cb0; text-decoration: none; }
  nav a:hover { text-decoration: underline; }
  .panel { background: white; padding: 1.5em; border-radius: 10px;
           box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 1.5em; }
  label { display: inline-block; margin-right: 1em; }
  input[type=number] { width: 80px; padding: 4px; }
  button { background: #2b6cb0; color: white; border: none;
           padding: 0.7em 1.5em; border-radius: 6px; cursor: pointer;
           font-size: 1em; }
  button:hover { background: #2c5282; }
  img { max-width: 100%; border-radius: 8px; }
  .meta { color: #555; font-size: 0.9em; }
  .note { color: #555; font-size: 0.9em; font-style: italic;
          margin-top: 0.5em; }
</style>
</head>
<body>
<h1>icon-lite — сравнение 1D и 2D</h1>
<nav>
  <a href="/">1D вертикальная колонка</a>
  <a href="/2d">2D конвекция</a>
  <a href="/compare">Сравнение 1D/2D</a>
</nav>

<div class="panel">
  <p>Одинаковые начальные условия (d&#952;, qv, z пузыря, dt).
     Число шагов уменьшено для скорости.</p>
  <form method="post">
    <label>d&#952;, K: <input type="number" name="dtheta" value="{{ dtheta }}" step="0.5" min="0" max="20"></label>
    <label>qv, г/кг: <input type="number" name="qv_max" value="{{ qv_max }}" step="1" min="0" max="30"></label>
    <label>z пузыря, км: <input type="number" name="zbubble" value="{{ zbubble }}" step="0.5" min="1" max="15"></label>
    <label>dt, с: <input type="number" name="dt" value="{{ dt }}" step="0.05" min="0.05" max="2"></label>
    <br><br>
    <button type="submit">Запустить сравнение</button>
  </form>
</div>

{% if image_1d and image_2d %}
<div class="panel">
  <h2>1D — вертикальная колонка</h2>
  <p class="meta">max|w| = {{ max_w_1d }} м/с</p>
  <img src="data:image/png;base64,{{ image_1d }}" alt="1D">
</div>

<div class="panel">
  <h2>2D — конвекция в плоскости (x, z)</h2>
  <p class="meta">max|w| = {{ max_w_2d }} м/с</p>
  <img src="data:image/png;base64,{{ image_2d }}" alt="2D">
</div>

<div class="panel">
  <h2>Физическая интерпретация</h2>
  <p>1D-модель изолирует вертикальное движение. 2D-модель включает
     горизонтальное растекание, которое концентрирует плавучесть
     и даёт более сильные восходящие потоки. Термодинамика в обеих
     моделях идентична.</p>
  <p class="note">Разница в max|w| — не ошибка, а физика:
     2D захватывает больше энергии за счёт горизонтальной сходимости.</p>
</div>
{% endif %}
</body>
</html>
'''


def run_model_1d(nlev, dtheta, qv_max, zbubble, dt, nsteps, use_diffusion,
                 return_profiles=False, dark=False):
    grid = make_vertical_grid(nlev=nlev, z_top=20000.0, stretch=1.15)
    z_mc, z_ifc = grid["z_mc"], grid["z_ifc"]
    wgtfac_c, inv_dz = grid["wgtfac_c"], 1.0 / grid["dz"]
    dz = grid["dz"]

    ref = make_reference_atmosphere(z_mc, z_ifc, theta_surf=300.0)
    rho = ref["rho_ref"].copy()
    theta_ref = ref["theta_ref"]
    exner_ref = ref["exner_ref"]

    shape = np.exp(-((z_mc - zbubble * 1000.0) / 1500.0) ** 2)
    theta = theta_ref + dtheta * shape
    qv = (qv_max / 1000.0) * shape
    qc = np.zeros_like(theta)
    qr = np.zeros_like(theta)
    w = np.zeros(len(z_ifc))

    n_snap = 8
    snap_every = max(1, nsteps // n_snap)
    times = [0.0]; w_hist = [w.copy()]; qv_hist = [qv.copy()]; qc_hist = [qc.copy()]; qr_hist = [qr.copy()]

    for n in range(nsteps):
        out = step_forward(rho, w, theta, qv, qc, qr, exner_ref, theta_ref,
                           wgtfac_c, inv_dz, dz, z_mc, z_ifc, dt,
                           use_diffusion=use_diffusion, dx=500.0)
        rho, w = out["rho"], out["w"]
        theta, qv, qc = out["theta"], out["qv"], out["qc"]
        qr = out["qr"]
        if (n + 1) % snap_every == 0 and len(times) < n_snap:
            times.append((n + 1) * dt)
            w_hist.append(w.copy()); qv_hist.append(qv.copy()); qc_hist.append(qc.copy()); qr_hist.append(qr.copy())

    from animation import make_snapshots_frame
    image_b64 = make_snapshots_frame(w_hist, qv_hist, qc_hist, qr_hist,
                                     z_ifc, z_mc, times, dark=dark)

    # w на интерфейсах (nlev+1) -> интерполируем на центры (nlev) для CSV
    w_mc = 0.5 * (w[:-1] + w[1:])

    out = {
        "image": image_b64,
        "max_w": float(np.max(np.abs(w))),
        "max_qv": float(np.max(qv)) * 1000,
        "max_qc": float(np.max(qc)) * 1000,
        "max_qr": float(np.max(qr)) * 1000,
        "max_dtheta": float(np.max(theta - theta_ref)),
    }
    if return_profiles:
        out.update({
            "z_mc": z_mc,
            "w_mc": w_mc,
            "theta": theta,
            "qv": qv,
            "qc": qc,
            "qr": qr,
        })
    return out


@app.route("/", methods=["GET", "POST"])
def index():
    params = {"nlev": 40, "dtheta": 2.0, "qv_max": 20.0,
              "zbubble": 5.0, "dt": 0.5, "nsteps": 1200,
              "use_diffusion": True}
    result = None
    if request.method == "POST":
        for k in params:
            if k == "use_diffusion":
                params[k] = request.form.get(k) is not None
            else:
                v = request.form.get(k)
                if v is None or v.strip() == "":
                    continue  # оставить дефолт из params
                try:
                    params[k] = int(v) if k in ("nlev", "nsteps") else float(v)
                except (TypeError, ValueError):
                    continue  # некорректное значение -> дефолт
        result = run_model_1d(**params)  # графики всегда светлые
    return render_template(
        "1d.html",
        active="1d",
        image=result["image"] if result else None,
        max_w=result["max_w"] if result else None,
        max_qv=result["max_qv"] if result else None,
        max_qc=result["max_qc"] if result else None,
        max_qr=result["max_qr"] if result else None,
        max_dtheta=result["max_dtheta"] if result else None,
        **params,
    )


def run_model_2d(nz, nx, x_total, dtheta, qv_max, zbubble, radius_x, dt, nsteps,
                 dark=False, return_profiles=False):
    g = make_grid2d(nz=nz, nx=nx, x_total=x_total * 1000.0, z_top=20000.0)
    init = make_initial_2d(
        g, None, qv_max=qv_max / 1000.0, dtheta=dtheta,
        x0_km=x_total / 2.0, z0_km=zbubble,
        radius_x_km=radius_x, radius_z_km=1.5,
    )
    theta_v_ref = virtual_theta(init["theta_ref_2d"], np.zeros_like(init["qv"]))
    exner_ref_2d = diagnose_exner(init["rho"], theta_v_ref)

    rho, w, vn = init["rho"].copy(), init["w"].copy(), init["vn"].copy()
    theta, qv, qc = init["theta"].copy(), init["qv"].copy(), init["qc"].copy()
    qr = np.zeros_like(qc)

    n_snap = 8
    snap_every = max(1, nsteps // n_snap)
    times = [0.0]; w_hist = [w.copy()]; qc_hist = [qc.copy()]; qr_hist = [qr.copy()]

    for n in range(nsteps):
        out = step_forward_2d(rho, w, vn, theta, qv, qc, qr,
                              exner_ref_2d, init["theta_ref_2d"], g, dt)
        rho, w, vn = out["rho"], out["w"], out["vn"]
        theta, qv, qc = out["theta"], out["qv"], out["qc"]
        qr = out["qr"]
        if (n + 1) % snap_every == 0 and len(times) < n_snap:
            times.append((n + 1) * dt)
            w_hist.append(w.copy()); qc_hist.append(qc.copy()); qr_hist.append(qr.copy())

    image_b64 = make_snapshots_2d(w_hist, qc_hist, qr_hist,
                                  g["x_mc"], g["z_ifc"], g["z_mc"], times,
                                  dark=dark)
    from plotting2d import make_snapshots_2d_3d
    image_3d_b64 = make_snapshots_2d_3d(w_hist, qc_hist, qr_hist,
                                        g["x_mc"], g["z_ifc"], g["z_mc"], times,
                                        dark=dark)

    # w на интерфейсах (nz+1,) -> интерполируем на центры (nz,) для CSV
    w_mc = 0.5 * (w[:-1] + w[1:])

    out = {
        "image": image_b64,
        "image_3d": image_3d_b64,
        "max_w": float(np.max(np.abs(w))),
        "max_vn": float(np.max(np.abs(vn))),
        "max_qc": float(np.max(qc)) * 1000,
        "max_qr": float(np.max(qr)) * 1000,
    }
    if return_profiles:
        out.update({
            "x_mc": g["x_mc"],
            "z_mc": g["z_mc"],
            "w_mc": w_mc,
            "theta": theta,
            "qv": qv,
            "qc": qc,
            "qr": qr,
        })
    return out


def run_model_1d_temperature(nlev, dtheta, qv_max, zbubble, dt, nsteps,
                             use_diffusion, return_profiles=False):
    """
    1D-симуляция с диагностикой температуры T = theta * Pi.
    Возвращает: image (профили T(z)), метрики T, и (опц.) профили для CSV.
    """
    grid = make_vertical_grid(nlev=nlev, z_top=20000.0, stretch=1.15)
    z_mc, z_ifc = grid["z_mc"], grid["z_ifc"]
    wgtfac_c, inv_dz = grid["wgtfac_c"], 1.0 / grid["dz"]
    dz = grid["dz"]

    ref = make_reference_atmosphere(z_mc, z_ifc, theta_surf=300.0)
    rho = ref["rho_ref"].copy()
    theta_ref = ref["theta_ref"]
    exner_ref = ref["exner_ref"]

    shape = np.exp(-((z_mc - zbubble * 1000.0) / 1500.0) ** 2)
    theta = theta_ref + dtheta * shape
    qv = (qv_max / 1000.0) * shape
    qc = np.zeros_like(theta)
    qr = np.zeros_like(theta)
    w = np.zeros(len(z_ifc))

    n_snap = 8
    snap_every = max(1, nsteps // n_snap)
    times = [0.0]
    theta_hist = [theta.copy()]
    exner_hist = [exner_ref.copy()]
    qc_hist = [qc.copy()]

    for n in range(nsteps):
        out = step_forward(rho, w, theta, qv, qc, qr, exner_ref, theta_ref,
                           wgtfac_c, inv_dz, dz, z_mc, z_ifc, dt,
                           use_diffusion=use_diffusion, dx=500.0)
        rho, w = out["rho"], out["w"]
        theta, qv, qc = out["theta"], out["qv"], out["qc"]
        qr = out["qr"]
        exner_now = out["exner"]

        if (n + 1) % snap_every == 0 and len(times) < n_snap:
            times.append((n + 1) * dt)
            theta_hist.append(theta.copy())
            exner_hist.append(exner_now.copy())
            qc_hist.append(qc.copy())

    image_b64 = make_snapshots_temperature(theta_hist, exner_hist, z_mc, times)

    T_K = diagnose_temperature(theta, exner_now)
    T_C = temperature_celsius(T_K)

    metrics = extract_t_metrics(theta, exner_now, z_mc, qc)

    out = {
        "image": image_b64,
        "T_surface_C": metrics["T_surface_C"],
        "T_2m_C": metrics["T_2m_C"],
        "T_cloud_C": metrics["T_cloud_C"],
        "T_min_C": metrics["T_min_C"],
        "T_max_C": metrics["T_max_C"],
    }
    if return_profiles:
        out.update({
            "z_mc": z_mc,
            "T_C": T_C,
            "theta": theta,
            "exner": exner_now,
            "qc": qc,
        })
    return out


@app.route("/temperature", methods=["GET", "POST"])
def temperature_route():
    params = {"nlev": 40, "dtheta": 2.0, "qv_max": 20.0,
              "zbubble": 5.0, "dt": 0.5, "nsteps": 1200,
              "use_diffusion": True}
    result = None
    if request.method == "POST":
        for k in params:
            if k == "use_diffusion":
                params[k] = request.form.get(k) is not None
            else:
                v = request.form.get(k)
                if v is None or v.strip() == "":
                    continue
                try:
                    params[k] = int(v) if k in ("nlev", "nsteps") else float(v)
                except (TypeError, ValueError):
                    continue
        result = run_model_1d_temperature(**params)
    return render_template(
        "temperature.html",
        image=result["image"] if result else None,
        T_surface=round(result["T_surface_C"], 2) if result else None,
        T_2m=round(result["T_2m_C"], 2) if result else None,
        T_cloud=round(result["T_cloud_C"], 2) if (result and result["T_cloud_C"] is not None) else None,
        T_min=round(result["T_min_C"], 2) if result else None,
        T_max=round(result["T_max_C"], 2) if result else None,
        active="temperature",
        **params,
    )


@app.route("/download/temperature")
def download_temperature():
    def _get_int(name, default):
        v = request.args.get(name)
        if v is None or v.strip() == "":
            return default
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    def _get_float(name, default):
        v = request.args.get(name)
        if v is None or v.strip() == "":
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    params = {
        "nlev": _get_int("nlev", 40),
        "dtheta": _get_float("dtheta", 2.0),
        "qv_max": _get_float("qv_max", 20.0),
        "zbubble": _get_float("zbubble", 5.0),
        "dt": _get_float("dt", 0.5),
        "nsteps": _get_int("nsteps", 1200),
        "use_diffusion": request.args.get("use_diffusion", "1") == "1",
    }
    res = run_model_1d_temperature(**params, return_profiles=True)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["z_m", "T_C", "theta_K", "exner", "qc_kgkg"])
    for i in range(len(res["z_mc"])):
        writer.writerow([
            f"{res['z_mc'][i]:.2f}",
            f"{res['T_C'][i]:.4f}",
            f"{res['theta'][i]:.6f}",
            f"{res['exner'][i]:.8f}",
            f"{res['qc'][i]:.8f}",
        ])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition":
                 "attachment; filename=icon_lite_temperature.csv"},
    )


@app.route("/storm", methods=["GET", "POST"])
def storm_route():
    params = {"nlev": 40, "dtheta": 2.0, "qv_max": 20.0,
              "zbubble": 5.0, "dt": 0.5, "nsteps": 1200,
              "use_diffusion": True}
    result = None
    if request.method == "POST":
        for k in params:
            if k == "use_diffusion":
                params[k] = request.form.get(k) is not None
            else:
                v = request.form.get(k)
                if v is None or v.strip() == "":
                    continue
                try:
                    params[k] = int(v) if k in ("nlev", "nsteps") else float(v)
                except (TypeError, ValueError):
                    continue
        result = run_model_1d_storm(**params, dark=_is_dark())
    return render_template(
        "storm.html",
        image=result["image"] if result else None,
        image_ts=result["image_ts"] if result else None,
        cape=round(result["cape_Jkg"], 1) if result else None,
        li=round(result["li_K"], 2) if (result and result["li_K"] is not None) else None,
        lfc=round(result["lfc_m"], 0) if (result and result["lfc_m"] is not None) else None,
        el=round(result["el_m"], 0) if (result and result["el_m"] is not None) else None,
        z_lcl=round(result["z_lcl_m"], 0) if result else None,
        max_w=round(result["max_w"], 2) if result else None,
        traffic=result["traffic"] if result else None,
        active="storm",
        **params,
    )


@app.route("/download/storm")
def download_storm():
    def _get_int(name, default):
        v = request.args.get(name)
        if v is None or v.strip() == "":
            return default
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    def _get_float(name, default):
        v = request.args.get(name)
        if v is None or v.strip() == "":
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    params = {
        "nlev": _get_int("nlev", 40),
        "dtheta": _get_float("dtheta", 2.0),
        "qv_max": _get_float("qv_max", 20.0),
        "zbubble": _get_float("zbubble", 5.0),
        "dt": _get_float("dt", 0.5),
        "nsteps": _get_int("nsteps", 1200),
        "use_diffusion": request.args.get("use_diffusion", "1") == "1",
    }
    res = run_model_1d_storm(**params, return_profiles=True)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["z_m", "theta_v_env_K", "theta_v_part_K",
                     "theta_K", "qv_kgkg", "exner", "CAPE_contrib"])
    for i in range(len(res["z_mc"])):
        writer.writerow([
            f"{res['z_mc'][i]:.2f}",
            f"{res['theta_v_env'][i]:.6f}",
            f"{res['theta_v_part'][i]:.6f}",
            f"{res['theta'][i]:.6f}",
            f"{res['qv'][i]:.8f}",
            f"{res['exner'][i]:.8f}",
            f"{res['cape_profile'][i]:.6f}",
        ])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition":
                 "attachment; filename=icon_lite_storm.csv"},
    )


@app.route("/download/storm.png")
def download_storm_png():
    def _get_int(name, default):
        v = request.args.get(name)
        if v is None or v.strip() == "":
            return default
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    def _get_float(name, default):
        v = request.args.get(name)
        if v is None or v.strip() == "":
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    params = {
        "nlev": _get_int("nlev", 40),
        "dtheta": _get_float("dtheta", 2.0),
        "qv_max": _get_float("qv_max", 20.0),
        "zbubble": _get_float("zbubble", 5.0),
        "dt": _get_float("dt", 0.5),
        "nsteps": _get_int("nsteps", 1200),
        "use_diffusion": request.args.get("use_diffusion", "1") == "1",
    }
    res = run_model_1d_storm(**params, return_png=True)
    return Response(
        res["png_bytes"],
        mimetype="image/png",
        headers={"Content-Disposition":
                 "attachment; filename=icon_lite_storm.png"},
    )


def run_model_2d_storm(nz, nx, x_total, dtheta, qv_max,
                       zbubble, radius_x, dt, nsteps,
                       use_diffusion=True, dark=False,
                       return_profiles=False):
    """
    2D simulation + storm diagnostics (CAPE in each column).

    Returns dict with:
        image    : base64 PNG of CAPE(x, z) map
        image_pr : base64 PNG of horizontal CAPE(x) profile
        cape_max, cape_mean, cape_min : float
        argmax_x : int
        max_w    : float
        traffic  : dict
        lfc_min, el_max : float or None
    """
    from grid2d import make_grid2d
    from state2d import make_initial_2d
    from state import diagnose_exner
    from thermo import virtual_theta
    from timestep2d import step_forward_2d

    g = make_grid2d(nz=nz, nx=nx, x_total=x_total * 1000.0,
                    z_top=20000.0, stretch=1.15)
    z_mc = g["z_mc"]
    x_mc = g["x_mc"]

    init = make_initial_2d(g, None, qv_max=qv_max / 1000.0,
                           dtheta=dtheta,
                           x0_km=x_total / 2.0, z0_km=zbubble,
                           radius_x_km=radius_x, radius_z_km=1.5)

    theta_ref_2d = init["theta_ref_2d"]
    theta_v_ref_2d = virtual_theta(theta_ref_2d,
                                   np.zeros_like(init["qv"]))
    exner_ref_2d = diagnose_exner(init["rho"], theta_v_ref_2d)

    rho = init["rho"].copy()
    w = init["w"].copy()
    vn = init["vn"].copy()
    theta = init["theta"].copy()
    qv = init["qv"].copy()
    qc = init["qc"].copy()
    qr = np.zeros_like(qc)

    exner_now = exner_ref_2d
    for n in range(nsteps):
        out = step_forward_2d(rho, w, vn, theta, qv, qc, qr,
                              exner_ref_2d, theta_ref_2d, g, dt)
        rho, w, vn = out["rho"], out["w"], out["vn"]
        theta, qv, qc = out["theta"], out["qv"], out["qc"]
        qr = out["qr"]
        exner_now = out.get("exner", exner_ref_2d)

    max_w = float(np.max(np.abs(w)))

    # --- CAPE по всем колонкам ---
    cape_res = compute_cape_2d(z_mc, theta, qv, exner_now)
    metrics = storm2d_metrics(cape_res, max_w)

    # --- артинки ---
    map_png = make_storm2d_map(x_mc, z_mc, cape_res["cape_2d"],
                               dark=dark,
                               cape_max=metrics["cape_max"],
                               cape_mean=metrics["cape_mean"],
                               lfc_x=cape_res["lfc_x"],
                               el_x=cape_res["el_x"],
                               traffic=metrics["traffic"])
    pr_png = make_storm2d_profile(x_mc, cape_res["cape_col"],
                                  dark=dark,
                                  traffic=metrics["traffic"])

    from plotting_storm2d import make_storm2d_map_3d
    map3d_png = make_storm2d_map_3d(x_mc, z_mc, cape_res["cape_2d"],
                                    dark=dark,
                                    cape_max=metrics["cape_max"],
                                    cape_mean=metrics["cape_mean"],
                                    lfc_x=cape_res["lfc_x"],
                                    el_x=cape_res["el_x"],
                                    traffic=metrics["traffic"])

    import base64 as _b64
    out_dict = {
        "image": _b64.b64encode(map_png).decode("ascii"),
        "image_pr": _b64.b64encode(pr_png).decode("ascii"),
        "image_3d": _b64.b64encode(map3d_png).decode("ascii"),
        "png_map_bytes": map_png,
        "png_profile_bytes": pr_png,
        "png_map3d_bytes": map3d_png,
        "cape_max": metrics["cape_max"],
        "cape_mean": metrics["cape_mean"],
        "cape_min": metrics["cape_min"],
        "argmax_x": int(cape_res["argmax_x"]),
        "lfc_min": metrics["lfc_min"],
        "el_max": metrics["el_max"],
        "max_w": max_w,
        "traffic": metrics["traffic"],
    }
    if return_profiles:
        out_dict.update({
            "z_mc": z_mc,
            "x_mc": x_mc,
            "cape_2d": cape_res["cape_2d"],
            "cape_col": cape_res["cape_col"],
            "lfc_x": cape_res["lfc_x"],
            "el_x": cape_res["el_x"],
            "theta": theta,
            "qv": qv,
            "exner": exner_now,
        })
    return out_dict


@app.route("/storm2d", methods=["GET", "POST"])
def storm2d_route():
    params = {"nz": 30, "nx": 40, "x_total": 20.0,
              "dtheta": 2.0, "qv_max": 20.0,
              "zbubble": 2.0, "radius_x": 4.0,
              "dt": 0.5, "nsteps": 200,
              "use_diffusion": True}
    result = None
    if request.method == "POST":
        for k in params:
            if k == "use_diffusion":
                params[k] = request.form.get(k) is not None
            else:
                v = request.form.get(k)
                if v is None or v.strip() == "":
                    continue
                try:
                    params[k] = int(v) if k in ("nz", "nx", "nsteps") else float(v)
                except (TypeError, ValueError):
                    continue
        result = run_model_2d_storm(**params, dark=_is_dark())
    return render_template(
        "storm2d.html",
        image=result["image"] if result else None,
        image_pr=result["image_pr"] if result else None,
        image_3d=result["image_3d"] if result else None,
        cape_max=round(result["cape_max"], 1) if result else None,
        cape_mean=round(result["cape_mean"], 1) if result else None,
        cape_min=round(result["cape_min"], 1) if result else None,
        argmax_x=result["argmax_x"] if result else None,
        lfc_min=round(result["lfc_min"], 0) if (result and result["lfc_min"] is not None) else None,
        el_max=round(result["el_max"], 0) if (result and result["el_max"] is not None) else None,
        max_w=round(result["max_w"], 2) if result else None,
        traffic=result["traffic"] if result else None,
        active="storm2d",
        **params,
    )


@app.route("/download/storm2d")
def download_storm2d():
    def _get_int(name, default):
        v = request.args.get(name)
        if v is None or v.strip() == "":
            return default
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    def _get_float(name, default):
        v = request.args.get(name)
        if v is None or v.strip() == "":
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    params = {
        "nz": _get_int("nz", 30),
        "nx": _get_int("nx", 40),
        "x_total": _get_float("x_total", 20.0),
        "dtheta": _get_float("dtheta", 2.0),
        "qv_max": _get_float("qv_max", 20.0),
        "zbubble": _get_float("zbubble", 2.0),
        "radius_x": _get_float("radius_x", 4.0),
        "dt": _get_float("dt", 0.5),
        "nsteps": _get_int("nsteps", 200),
        "use_diffusion": request.args.get("use_diffusion", "1") == "1",
    }
    res = run_model_2d_storm(**params, return_profiles=True)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["x_km", "z_km", "CAPE_contrib"])
    for j in range(len(res["x_mc"])):
        for i in range(len(res["z_mc"])):
            writer.writerow([
                f"{res['x_mc'][j] / 1000.0:.3f}",
                f"{res['z_mc'][i] / 1000.0:.3f}",
                f"{res['cape_2d'][i, j]:.4f}",
            ])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition":
                 "attachment; filename=icon_lite_storm2d.csv"},
    )


@app.route("/download/storm2d.png")
def download_storm2d_png():
    def _get_int(name, default):
        v = request.args.get(name)
        if v is None or v.strip() == "":
            return default
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    def _get_float(name, default):
        v = request.args.get(name)
        if v is None or v.strip() == "":
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    params = {
        "nz": _get_int("nz", 30),
        "nx": _get_int("nx", 40),
        "x_total": _get_float("x_total", 20.0),
        "dtheta": _get_float("dtheta", 2.0),
        "qv_max": _get_float("qv_max", 20.0),
        "zbubble": _get_float("zbubble", 2.0),
        "radius_x": _get_float("radius_x", 4.0),
        "dt": _get_float("dt", 0.5),
        "nsteps": _get_int("nsteps", 200),
        "use_diffusion": request.args.get("use_diffusion", "1") == "1",
    }
    res = run_model_2d_storm(**params)
    return Response(
        res["png_map_bytes"],
        mimetype="image/png",
        headers={"Content-Disposition":
                 "attachment; filename=icon_lite_storm2d_map.png"},
    )


@app.route("/download/storm2d.map3d.png")
def download_storm2d_map3d():
    def _get_int(name, default):
        v = request.args.get(name)
        if v is None or v.strip() == "":
            return default
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    def _get_float(name, default):
        v = request.args.get(name)
        if v is None or v.strip() == "":
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    params = {
        "nz": _get_int("nz", 30),
        "nx": _get_int("nx", 40),
        "x_total": _get_float("x_total", 20.0),
        "dtheta": _get_float("dtheta", 2.0),
        "qv_max": _get_float("qv_max", 20.0),
        "zbubble": _get_float("zbubble", 2.0),
        "radius_x": _get_float("radius_x", 4.0),
        "dt": _get_float("dt", 0.5),
        "nsteps": _get_int("nsteps", 200),
        "use_diffusion": request.args.get("use_diffusion", "1") == "1",
    }
    res = run_model_2d_storm(**params)
    return Response(
        res["png_map3d_bytes"],
        mimetype="image/png",
        headers={"Content-Disposition":
                 "attachment; filename=icon_lite_storm2d_map3d.png"},
    )


@app.route("/download/storm2d.profile.png")
def download_storm2d_profile_png():
    def _get_int(name, default):
        v = request.args.get(name)
        if v is None or v.strip() == "":
            return default
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    def _get_float(name, default):
        v = request.args.get(name)
        if v is None or v.strip() == "":
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    params = {
        "nz": _get_int("nz", 30),
        "nx": _get_int("nx", 40),
        "x_total": _get_float("x_total", 20.0),
        "dtheta": _get_float("dtheta", 2.0),
        "qv_max": _get_float("qv_max", 20.0),
        "zbubble": _get_float("zbubble", 2.0),
        "radius_x": _get_float("radius_x", 4.0),
        "dt": _get_float("dt", 0.5),
        "nsteps": _get_int("nsteps", 200),
        "use_diffusion": request.args.get("use_diffusion", "1") == "1",
    }
    res = run_model_2d_storm(**params)
    return Response(
        res["png_profile_bytes"],
        mimetype="image/png",
        headers={"Content-Disposition":
                 "attachment; filename=icon_lite_storm2d_profile.png"},
    )


@app.route("/models", methods=["GET"])
def models_hub():
    return render_template("models.html", active="models")


@app.route("/storm-hub", methods=["GET"])
def storm_hub():
    return render_template("storm_hub.html", active="storm-hub")


@app.route("/download/1d")
def download_1d():
    params = {
        "nlev": int(request.args.get("nlev", 40)),
        "dtheta": float(request.args.get("dtheta", 2.0)),
        "qv_max": float(request.args.get("qv_max", 20.0)),
        "zbubble": float(request.args.get("zbubble", 5.0)),
        "dt": float(request.args.get("dt", 0.5)),
        "nsteps": int(request.args.get("nsteps", 1200)),
        "use_diffusion": request.args.get("use_diffusion", "1") == "1",
    }
    res = run_model_1d(**params, return_profiles=True)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["z_m", "w_ms", "theta_K", "qv_kgkg", "qc_kgkg", "qr_kgkg"])
    for i in range(len(res["z_mc"])):
        writer.writerow([
            f"{res['z_mc'][i]:.2f}",
            f"{res['w_mc'][i]:.6f}",
            f"{res['theta'][i]:.6f}",
            f"{res['qv'][i]:.8f}",
            f"{res['qc'][i]:.8f}",
            f"{res['qr'][i]:.8f}",
        ])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition":
                 "attachment; filename=icon_lite_1d.csv"},
    )


@app.route("/download/2d")
def download_2d():
    def _get_int(name, default):
        v = request.args.get(name)
        if v is None or v.strip() == "":
            return default
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    def _get_float(name, default):
        v = request.args.get(name)
        if v is None or v.strip() == "":
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    params = {
        "nz": _get_int("nz", 40),
        "nx": _get_int("nx", 80),
        "x_total": _get_float("x_total", 40.0),
        "dtheta": _get_float("dtheta", 2.0),
        "qv_max": _get_float("qv_max", 20.0),
        "zbubble": _get_float("zbubble", 3.0),
        "radius_x": _get_float("radius_x", 5.0),
        "dt": _get_float("dt", 0.5),
        "nsteps": _get_int("nsteps", 800),
    }
    res = run_model_2d(**params, return_profiles=True)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["x_km", "z_km", "w_ms", "theta_K",
                     "qv_kgkg", "qc_kgkg", "qr_kgkg"])

    x_km = res["x_mc"] / 1000.0
    z_km = res["z_mc"] / 1000.0
    for i in range(len(x_km)):
        for k in range(len(z_km)):
            writer.writerow([
                f"{x_km[i]:.4f}",
                f"{z_km[k]:.4f}",
                f"{res['w_mc'][k, i]:.6f}",
                f"{res['theta'][k, i]:.6f}",
                f"{res['qv'][k, i]:.8f}",
                f"{res['qc'][k, i]:.8f}",
                f"{res['qr'][k, i]:.8f}",
            ])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition":
                 "attachment; filename=icon_lite_2d.csv"},
    )


@app.route("/download/2d.3d.png")
def download_2d_3d():
    import base64 as _b64

    def _get_int(name, default):
        v = request.args.get(name)
        if v is None or v.strip() == "":
            return default
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    def _get_float(name, default):
        v = request.args.get(name)
        if v is None or v.strip() == "":
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    params = {
        "nz": _get_int("nz", 40),
        "nx": _get_int("nx", 80),
        "x_total": _get_float("x_total", 40.0),
        "dtheta": _get_float("dtheta", 2.0),
        "qv_max": _get_float("qv_max", 20.0),
        "zbubble": _get_float("zbubble", 3.0),
        "radius_x": _get_float("radius_x", 5.0),
        "dt": _get_float("dt", 0.5),
        "nsteps": _get_int("nsteps", 800),
    }
    res = run_model_2d(**params)
    png_bytes = _b64.b64decode(res["image_3d"])
    return Response(
        png_bytes,
        mimetype="image/png",
        headers={"Content-Disposition":
                 "attachment; filename=icon_lite_2d_snapshots_3d.png"},
    )


@app.route("/compare", methods=["GET", "POST"])
def compare():
    params = {"dtheta": 2.0, "qv_max": 20.0, "zbubble": 3.0, "dt": 0.5}
    result_1d = result_2d = None
    if request.method == "POST":
        for k in params:
            v = request.form.get(k)
            if v is None or v.strip() == "":
                continue
            try:
                params[k] = float(v)
            except (TypeError, ValueError):
                continue

        result_1d = run_model_1d(
            nlev=40,
            dtheta=params["dtheta"],
            qv_max=params["qv_max"],
            zbubble=params["zbubble"],
            dt=params["dt"],
            nsteps=600,
            use_diffusion=True,
        )

        result_2d = run_model_2d(
            nz=40, nx=80, x_total=40.0,
            dtheta=params["dtheta"],
            qv_max=params["qv_max"],
            zbubble=params["zbubble"],
            radius_x=5.0,
            dt=params["dt"],
            nsteps=400,
        )

    return render_template(
        "compare.html",
        active="compare",
        image_1d=result_1d["image"] if result_1d else None,
        image_2d=result_2d["image"] if result_2d else None,
        max_w_1d=result_1d["max_w"] if result_1d else None,
        max_w_2d=result_2d["max_w"] if result_2d else None,
        **params,
    )


@app.route("/2d", methods=["GET", "POST"])
def index_2d():
    params = {
        "nz": 40, "nx": 80, "x_total": 40.0, "dtheta": 2.0,
        "qv_max": 20.0, "zbubble": 3.0, "radius_x": 5.0,
        "dt": 0.5, "nsteps": 800,
    }
    result = None
    if request.method == "POST":
        for k in params:
            v = request.form.get(k)
            if v is None or v.strip() == "":
                continue  # оставить дефолт
            try:
                params[k] = int(v) if k in ("nz", "nx", "nsteps") else float(v)
            except (TypeError, ValueError):
                continue
        result = run_model_2d(**params)  # графики всегда светлые
    return render_template(
        "2d.html",
        active="2d",
        image=result["image"] if result else None,
        image_3d=result["image_3d"] if result else None,
        max_w=result["max_w"] if result else None,
        max_vn=result["max_vn"] if result else None,
        max_qc=result["max_qc"] if result else None,
        max_qr=result["max_qr"] if result else None,
        **params,
    )


def run_model_1d_fog(nlev, dtheta, qv_max, zbubble, dt, nsteps,
                     k_rad, use_diffusion, return_profiles=False):
    """
    Радиационный туман в 1D: приземное выхолаживание + конденсация.
    Возвращает: image (профиль qc(z)), метрики тумана + T.
    """
    grid = make_vertical_grid(nlev=nlev, z_top=20000.0, stretch=1.15)
    z_mc, z_ifc = grid["z_mc"], grid["z_ifc"]
    wgtfac_c, inv_dz = grid["wgtfac_c"], 1.0 / grid["dz"]
    dz = grid["dz"]

    ref = make_reference_atmosphere(z_mc, z_ifc, theta_surf=300.0)
    rho = ref["rho_ref"].copy()
    theta_ref = ref["theta_ref"]
    exner_ref = ref["exner_ref"]

    # Ночная инверсия: тонкий слой у поверхности чуть холоднее + высокая влажность
    shape = np.exp(-((z_mc - zbubble * 1000.0) / 1500.0) ** 2)
    theta = theta_ref + dtheta * shape
    # Приземный слой: понижаем theta у первых 2 уровней (инверсия)
    theta[:2] = theta_ref[:2] - 2.0
    qv = (qv_max / 1000.0) * np.ones_like(theta)   # влажный слой снизу
    qv[10:] = (qv_max / 1000.0) * 0.3              # выше — суше
    qc = np.zeros_like(theta)
    qr = np.zeros_like(theta)
    w = np.zeros(len(z_ifc))

    n_snap = 8
    snap_every = max(1, nsteps // n_snap)
    times = [0.0]
    qc_hist = [qc.copy()]
    theta_hist = [theta.copy()]
    exner_hist = [exner_ref.copy()]

    for n in range(nsteps):
        out = step_forward(rho, w, theta, qv, qc, qr, exner_ref, theta_ref,
                           wgtfac_c, inv_dz, dz, z_mc, z_ifc, dt,
                           use_diffusion=use_diffusion, dx=500.0,
                           radiative_cooling=True, k_rad=k_rad)
        rho, w = out["rho"], out["w"]
        theta, qv, qc = out["theta"], out["qv"], out["qc"]
        qr = out["qr"]
        exner_now = out["exner"]

        if (n + 1) % snap_every == 0 and len(times) < n_snap:
            times.append((n + 1) * dt)
            qc_hist.append(qc.copy())
            theta_hist.append(theta.copy())
            exner_hist.append(exner_now.copy())

    # График — эволюция профиля qc(z)
    from plotting_fog import make_snapshots_fog_1d
    image_b64 = make_snapshots_fog_1d(qc_hist, z_mc, times)

    fm = fog_metrics(qc, z_mc)
    T_K = diagnose_temperature(theta, exner_now)
    T_C = temperature_celsius(T_K)

    out = {
        "image": image_b64,
        "has_fog": fm["has_fog"],
        "vis_min_m": fm["vis_min_m"],
        "qc_surface_gkg": fm["qc_surface_gkg"],
        "top_fog_m": fm["top_fog_m"],
        "T_surface_C": float(T_C[0]),
        "T_2m_C": float(temperature_celsius(
            extrapolate_t_2m(T_K, z_mc, z_target=2.0))),
    }
    if return_profiles:
        out.update({
            "z_mc": z_mc,
            "qc": qc,
            "T_C": T_C,
            "vis_m": visibility_from_qc(qc),
        })
    return out


def run_model_2d_fog(nz, nx, x_total, dtheta, qv_max, zbubble,
                     radius_x, dt, nsteps, U, k_surf, theta_surf_offset):
    """
    двективный туман в 2D: тёплый влажный воздух над холодной поверхностью.
    озвращает: image (поле qc(x, z)), метрики тумана.
    """
    g = make_grid2d(nz=nz, nx=nx, x_total=x_total * 1000.0, z_top=20000.0)
    z_mc = g["z_mc"]

    # Создаём базовую референсную стратификацию
    init = make_initial_2d(
        g, None, qv_max=qv_max / 1000.0, dtheta=0.0,
        x0_km=x_total / 2.0, z0_km=zbubble,
        radius_x_km=radius_x, radius_z_km=1.5,
    )
    theta_ref_2d = init["theta_ref_2d"]

    # --- равильное начальное условие: тёплый влажный слой внизу ---
    nz_arr = z_mc.shape[0]
    # лажность: qv_max в нижних 2 км, потом экспоненциально падает
    qv_prof = np.where(z_mc < 2000.0,
                       qv_max / 1000.0,
                       (qv_max / 1000.0) * np.exp(-(z_mc - 2000.0) / 1500.0))
    # Тёплый слой: theta_ref + dtheta в нижних 2 км
    theta_prof = np.where(z_mc < 2000.0,
                          theta_ref_2d[:, 0] + dtheta,
                          theta_ref_2d[:, 0] + dtheta * np.exp(-(z_mc - 2000.0) / 1500.0))

    theta = np.tile(theta_prof[:, None], (1, nx))
    qv = np.tile(qv_prof[:, None], (1, nx))
    qc = np.zeros_like(theta)
    qr = np.zeros_like(theta)

    # идростатика: rho = референсная плотность
    rho = init["rho"].copy()

    # Скорости — нули
    w = np.zeros((nz + 1, nx))
    vn = np.zeros((nz, nx + 1))

    # иагностика Exner
    theta_v_ref = virtual_theta(theta_ref_2d, np.zeros_like(qv))
    exner_ref_2d = diagnose_exner(rho, virtual_theta(theta, qv))

    # Холодная поверхность
    theta_surf = float(theta[0, 0]) - theta_surf_offset

    # Слой тумана — до 200 м
    k_h = int(np.searchsorted(z_mc, 200.0, side="right"))
    if k_h < 1:
        k_h = 1

    n_snap = 8
    snap_every = max(1, nsteps // n_snap)
    times = [0.0]
    qc_hist = [qc.copy()]
    w_hist = [w.copy()]

    for n in range(nsteps):
        # Шаг динамики + физики
        out = step_forward_2d(rho, w, vn, theta, qv, qc, qr,
                              exner_ref_2d, theta_ref_2d, g, dt)
        rho, w, vn = out["rho"], out["w"], out["vn"]
        theta, qv, qc = out["theta"], out["qv"], out["qc"]
        qr = out["qr"]

        # риземное охлаждение в слое до 200 м
        for k in range(k_h):
            theta[k, :] = theta[k, :] - k_surf * U * (theta[k, :] - theta_surf) * dt

        if (n + 1) % snap_every == 0 and len(times) < n_snap:
            times.append((n + 1) * dt)
            qc_hist.append(qc.copy())
            w_hist.append(w.copy())

    from plotting_fog import make_snapshots_fog_2d
    image_b64 = make_snapshots_fog_2d(qc_hist,
                                      g["x_mc"], g["z_ifc"], g["z_mc"], times)

    fm = fog_metrics(qc, g["z_mc"])

    return {
        "image": image_b64,
        "has_fog": fm["has_fog"],
        "vis_min_m": fm["vis_min_m"],
        "qc_surface_gkg": fm["qc_surface_gkg"],
        "top_fog_m": fm["top_fog_m"],
    }


@app.route("/fog", methods=["GET", "POST"])
def fog_route():
    # Параметры 1D (радиационный)
    params_1d = {"nlev": 40, "dtheta": 2.0, "qv_max": 20.0,
                 "zbubble": 5.0, "dt": 0.5, "nsteps": 600,
                 "k_rad": 1.0e-4, "use_diffusion": True}
    # Параметры 2D (адвективный)
    params_2d = {"nz": 40, "nx": 80, "x_total": 40.0, "dtheta": 2.0,
                 "qv_max": 20.0, "zbubble": 3.0, "radius_x": 5.0,
                 "dt": 0.5, "nsteps": 600,
                 "U": 5.0, "k_surf": 5.0e-4, "theta_surf_offset": 3.0}

    mode = "radiative"
    result = None

    if request.method == "POST":
        mode = request.form.get("mode", "radiative")
        if mode == "advective":
            for k in params_2d:
                v = request.form.get(k)
                if v is None or v.strip() == "":
                    continue
                try:
                    params_2d[k] = int(v) if k in ("nz", "nx", "nsteps") else float(v)
                except (TypeError, ValueError):
                    continue
            result = run_model_2d_fog(**params_2d)
        else:
            for k in params_1d:
                if k == "use_diffusion":
                    params_1d[k] = request.form.get(k) is not None
                else:
                    v = request.form.get(k)
                    if v is None or v.strip() == "":
                        continue
                    try:
                        params_1d[k] = int(v) if k in ("nlev", "nsteps") else float(v)
                    except (TypeError, ValueError):
                        continue
            result = run_model_1d_fog(**params_1d)

    # Объединяем params_2d и params_1d без конфликта общих ключей
    merged = dict(params_2d)
    merged.update(params_1d)   # 1D-версия перекрывает общие

    return render_template(
        "fog.html",
        active="fog",
        mode=mode,
        image=result["image"] if result else None,
        has_fog=result["has_fog"] if result else None,
        vis_min_m=round(result["vis_min_m"], 1) if result else None,
        qc_surface_gkg=round(result["qc_surface_gkg"], 3) if result else None,
        top_fog_m=round(result["top_fog_m"], 1) if result else None,
        **merged,
    )


def run_model_1d_storm(nlev, dtheta, qv_max, zbubble, dt, nsteps,
                        use_diffusion, return_profiles=False, dark=False,
                        return_png=False):
    """
    1D-симуляция + грозовая диагностика (CAPE, LI, светофор).

    озвращает: image (профиль θ_v с CAPE), метрики, светофор.
    """
    grid = make_vertical_grid(nlev=nlev, z_top=20000.0, stretch=1.15)
    z_mc, z_ifc = grid["z_mc"], grid["z_ifc"]
    wgtfac_c, inv_dz = grid["wgtfac_c"], 1.0 / grid["dz"]
    dz = grid["dz"]

    ref = make_reference_atmosphere(z_mc, z_ifc, theta_surf=300.0,
                                z_trop=11000.0,
                                dtheta_dz_strat=2.0e-3)
    rho = ref["rho_ref"].copy()
    theta_ref = ref["theta_ref"]
    exner_ref = ref["exner_ref"]

    # риземное возмущение: тёплый влажный слой у земли,
    # экспоненциально убывающий с высотой (типично для конвективного
    # пограничного слоя).
    shape = np.exp(-z_mc / (zbubble * 1000.0))
    theta = theta_ref + dtheta * shape
    qv = (qv_max / 1000.0) * shape
    qc = np.zeros_like(theta)
    qr = np.zeros_like(theta)
    w = np.zeros(len(z_ifc))

    exner_now = exner_ref
    times_hist = [0.0]
    w_max_hist = [float(np.max(np.abs(w)))]
    _snap_every = max(1, nsteps // 40)
    for n in range(nsteps):
        out = step_forward(rho, w, theta, qv, qc, qr, exner_ref, theta_ref,
                           wgtfac_c, inv_dz, dz, z_mc, z_ifc, dt,
                           use_diffusion=use_diffusion, dx=500.0)
        rho, w = out["rho"], out["w"]
        theta, qv, qc = out["theta"], out["qv"], out["qc"]
        qr = out["qr"]
        exner_now = out.get("exner", exner_ref)
        if (n + 1) % _snap_every == 0:
            times_hist.append((n + 1) * dt)
            w_max_hist.append(float(np.max(np.abs(w))))

    max_w = float(np.max(np.abs(w)))

    # --- розовая диагностика ---
    # ерём самую тёплую/влажную частицу в приземном слое (0-500 м).
    theta_v_env = virtual_theta(theta, qv)
    k_surf = int(np.searchsorted(z_mc, 500.0, side="right"))
    if k_surf < 1:
        k_surf = 1
    theta_part_surf = float(np.max(theta[:k_surf]))
    qv_part_surf = float(np.max(qv[:k_surf]))

    cape_info = compute_cape(z_mc, theta, qv, exner_now,
                             theta_part_surf=theta_part_surf,
                             qv_part_surf=qv_part_surf)
    li_K = compute_li(z_mc, theta, qv, exner_now,
                      theta_part_surf=theta_part_surf,
                      qv_part_surf=qv_part_surf)
    traffic = storm_index(cape_info["cape_Jkg"], li_K, max_w)

    image_ts_b64 = None
    if not return_png:
        ts_png = make_storm_timeseries(times_hist, w_max_hist, dark=dark,
                                       cape_Jkg=cape_info["cape_Jkg"],
                                       li_K=li_K if np.isfinite(li_K) else None,
                                       max_w=max_w)
        import base64 as _b64
        image_ts_b64 = _b64.b64encode(ts_png).decode("ascii")

    _profile_args = dict(
        z_mc=z_mc,
        theta_v_env=cape_info["theta_v_env"],
        theta_v_part=cape_info["theta_v_part"],
        cape_profile=cape_info["cape_profile"],
        lfc_m=cape_info["lfc_m"],
        el_m=cape_info["el_m"],
        z_lcl_m=cape_info["z_lcl_m"],
        z_trop_m=cape_info.get("z_trop_m"),
        dark=dark,
        cape_Jkg=cape_info["cape_Jkg"],
        li_K=li_K if np.isfinite(li_K) else 0.0,
        max_w=max_w,
        traffic=traffic,
    )
    png_bytes = None
    if return_png:
        png_bytes = make_storm_profile_bytes(**_profile_args)
        image_b64 = None
    else:
        image_b64 = make_storm_profile(**_profile_args)

    out = {
        "image": image_b64,
        "image_ts": image_ts_b64,
        "png_bytes": png_bytes,
        "cape_Jkg": float(cape_info["cape_Jkg"]),
        "li_K": float(li_K) if np.isfinite(li_K) else None,
        "lfc_m": cape_info["lfc_m"],
        "el_m": cape_info["el_m"],
        "z_lcl_m": cape_info["z_lcl_m"],
        "z_trop_m": cape_info.get("z_trop_m"),
        "max_w": max_w,
        "traffic": traffic,
    }
    if return_profiles:
        out.update({
            "z_mc": z_mc,
            "theta_v_env": cape_info["theta_v_env"],
            "theta_v_part": cape_info["theta_v_part"],
            "cape_profile": cape_info["cape_profile"],
            "theta": theta,
            "qv": qv,
            "exner": exner_now,
        })
    return out


@app.route("/download/fog")
def download_fog():
    def _get_int(name, default):
        v = request.args.get(name)
        if v is None or v.strip() == "":
            return default
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    def _get_float(name, default):
        v = request.args.get(name)
        if v is None or v.strip() == "":
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    mode = request.args.get("mode", "radiative")

    if mode == "radiative":
        params = {
            "nlev": _get_int("nlev", 40),
            "dtheta": _get_float("dtheta", 2.0),
            "qv_max": _get_float("qv_max", 20.0),
            "zbubble": _get_float("zbubble", 5.0),
            "dt": _get_float("dt", 0.5),
            "nsteps": _get_int("nsteps", 600),
            "k_rad": _get_float("k_rad", 1e-4),
            "use_diffusion": request.args.get("use_diffusion", "1") == "1",
        }
        res = run_model_1d_fog(**params, return_profiles=True)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["z_m", "qc_kgkg", "T_C", "vis_m"])
        for i in range(len(res["z_mc"])):
            writer.writerow([
                f"{res['z_mc'][i]:.2f}",
                f"{res['qc'][i]:.8f}",
                f"{res['T_C'][i]:.4f}",
                f"{res['vis_m'][i]:.1f}",
            ])
    else:
        # Адвективный — экспортируем как 2D-поле (x, z, qc)
        params = {
            "nz": _get_int("nz", 40),
            "nx": _get_int("nx", 80),
            "x_total": _get_float("x_total", 40.0),
            "dtheta": _get_float("dtheta", 2.0),
            "qv_max": _get_float("qv_max", 20.0),
            "zbubble": _get_float("zbubble", 3.0),
            "radius_x": _get_float("radius_x", 5.0),
            "dt": _get_float("dt", 0.5),
            "nsteps": _get_int("nsteps", 600),
            "U": _get_float("U", 5.0),
            "k_surf": _get_float("k_surf", 5e-4),
            "theta_surf_offset": _get_float("theta_surf_offset", 3.0),
        }
        res = run_model_2d_fog(**params)
        # Упрощённый экспорт — метаданные + последнее поле qc недоступно через res
        # Оставим CSV с метриками
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["metric", "value"])
        writer.writerow(["has_fog", res["has_fog"]])
        writer.writerow(["vis_min_m", res["vis_min_m"]])
        writer.writerow(["qc_surface_gkg", res["qc_surface_gkg"]])
        writer.writerow(["top_fog_m", res["top_fog_m"]])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition":
                 "attachment; filename=icon_lite_fog.csv"},
    )


# =====================================================================
# 3D bubble page
# =====================================================================

from app_3d_helper import run_model_3d, run_model_3d_rising


@app.route("/3d", methods=["GET", "POST"])
def route_3d():
    params = {
        "nx": 24, "ny": 24, "nz": 16,
        "x_total": 20.0, "y_total": 20.0, "z_top": 15.0,
        "dtheta": 2.0, "qv_max": 0.0, "z0": 3.0,
        "radius_x": 3.0, "radius_y": 3.0, "radius_z": 1.5,
        "dt": 0.5, "nsteps": 200,
        "mode": "static", "use_pressure": True,
    }
    result = None
    if request.method == "POST":
        for k in params:
            if k == "use_pressure":
                params[k] = request.form.get(k) is not None
            elif k == "mode":
                params[k] = request.form.get(k, "static")
            else:
                v = request.form.get(k)
                if v is None or v.strip() == "":
                    continue
                try:
                    if k in ("nx", "ny", "nz", "nsteps"):
                        params[k] = int(v)
                    else:
                        params[k] = float(v)
                except (TypeError, ValueError):
                    continue

        dark = _is_dark()
        if params["mode"] == "rising":
            result = run_model_3d_rising(
                nx=params["nx"], ny=params["ny"], nz=params["nz"],
                x_total=params["x_total"], y_total=params["y_total"],
                z_top=params["z_top"],
                dtheta=params["dtheta"], qv_max=params["qv_max"],
                z0=params["z0"],
                radius_x=params["radius_x"],
                radius_y=params["radius_y"],
                radius_z=params["radius_z"],
                dt=params["dt"], nsteps=params["nsteps"],
                use_pressure=params["use_pressure"],
                dark=dark,
            )
        else:
            result = run_model_3d(
                nx=params["nx"], ny=params["ny"], nz=params["nz"],
                x_total=params["x_total"], y_total=params["y_total"],
                z_top=params["z_top"],
                dtheta=params["dtheta"], qv_max=params["qv_max"],
                z0=params["z0"],
                radius_x=params["radius_x"],
                radius_y=params["radius_y"],
                radius_z=params["radius_z"],
                dark=dark,
            )

    ctx = {"active": "3d", **params}
    if result:
        ctx.update(result)
    return render_template("3d.html", **ctx)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)