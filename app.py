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

    # w на интерфейсах (nz+1,) -> интерполируем на центры (nz,) для CSV
    w_mc = 0.5 * (w[:-1] + w[1:])

    out = {
        "image": image_b64,
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
        max_w=result["max_w"] if result else None,
        max_vn=result["max_vn"] if result else None,
        max_qc=result["max_qc"] if result else None,
        max_qr=result["max_qr"] if result else None,
        **params,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)