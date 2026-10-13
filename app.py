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
from flask import Flask, render_template_string, request

from grid import make_vertical_grid
from grid2d import make_grid2d
from state import make_reference_atmosphere, diagnose_exner
from state2d import make_initial_2d
from timestep import step_forward
from timestep2d import step_forward_2d
from thermo import virtual_theta
from plotting2d import make_snapshots_2d

app = Flask(__name__)


# ====================== HTML шаблоны ======================

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
<nav><a href="/">1D вертикальная колонка</a><a href="/2d">2D конвекция</a></nav>
<div class="panel">
  <p>Модель вертикальной колонки атмосферы с влажностью и конденсацией.
     Зелёная кривая — облачная вода qc.</p>
  <form method="post">
    <label>Уровни: <input type="number" name="nlev" value="{{ nlev }}" min="10" max="100"></label>
    <label>dθ, K: <input type="number" name="dtheta" value="{{ dtheta }}" step="0.5" min="0" max="20"></label>
    <label>qv, г/кг: <input type="number" name="qv_max" value="{{ qv_max }}" step="1" min="0" max="30"></label>
    <label>z пузыря, км: <input type="number" name="zbubble" value="{{ zbubble }}" step="0.5" min="1" max="15"></label>
    <label>dt, с: <input type="number" name="dt" value="{{ dt }}" step="0.05" min="0.05" max="2"></label>
    <label>Шагов: <input type="number" name="nsteps" value="{{ nsteps }}" min="100" max="5000"></label>
    <br><br>
    <button type="submit">Запустить 1D</button>
  </form>
</div>
{% if image %}
<div class="panel">
  <h2>Эволюция конвективного облака (1D)</h2>
  <p class="meta">max|w| = {{ max_w }} м/с | max qv = {{ max_qv }} г/кг |
     max qc = {{ max_qc }} г/кг | max dθ = {{ max_dtheta }} K</p>
  <img src="data:image/png;base64,{{ image }}" alt="1D">
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
<nav><a href="/">1D вертикальная колонка</a><a href="/2d">2D конвекция</a></nav>
<div class="panel">
  <p>Двумерная модель в плоскости (x, z). Тёплый влажный пузырь
     всплывает, растекается, формирует облако. Красное — восходящий поток,
     синее — нисходящий. Зелёные контуры — облачная вода qc.</p>
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
     max qc = {{ max_qc }} г/кг</p>
  <img src="data:image/png;base64,{{ image }}" alt="2D">
</div>
{% endif %}
</body>
</html>
'''


# ====================== 1D ======================

def run_model_1d(nlev, dtheta, qv_max, zbubble, dt, nsteps):
    grid = make_vertical_grid(nlev=nlev, z_top=20000.0, stretch=1.15)
    z_mc, z_ifc = grid["z_mc"], grid["z_ifc"]
    wgtfac_c, inv_dz = grid["wgtfac_c"], 1.0 / grid["dz"]

    ref = make_reference_atmosphere(z_mc, z_ifc, theta_surf=300.0)
    rho = ref["rho_ref"].copy()
    theta_ref = ref["theta_ref"]
    exner_ref = ref["exner_ref"]

    shape = np.exp(-((z_mc - zbubble * 1000.0) / 1500.0) ** 2)
    theta = theta_ref + dtheta * shape
    qv = (qv_max / 1000.0) * shape
    qc = np.zeros_like(theta)
    w = np.zeros(len(z_ifc))

    n_snap = 8
    snap_every = max(1, nsteps // n_snap)
    times = [0.0]; w_hist = [w.copy()]; qv_hist = [qv.copy()]; qc_hist = [qc.copy()]

    for n in range(nsteps):
        out = step_forward(rho, w, theta, qv, qc, exner_ref, theta_ref,
                           wgtfac_c, inv_dz, dt)
        rho, w = out["rho"], out["w"]
        theta, qv, qc = out["theta"], out["qv"], out["qc"]
        if (n + 1) % snap_every == 0 and len(times) < n_snap:
            times.append((n + 1) * dt)
            w_hist.append(w.copy()); qv_hist.append(qv.copy()); qc_hist.append(qc.copy())

    # используем старую 1D-визуализацию — импортируем на месте
    from animation import make_snapshots_frame
    image_b64 = make_snapshots_frame(w_hist, qv_hist, qc_hist, z_ifc, z_mc, times)
    return {
        "image": image_b64,
        "max_w": float(np.max(np.abs(w))),
        "max_qv": float(np.max(qv)) * 1000,
        "max_qc": float(np.max(qc)) * 1000,
        "max_dtheta": float(np.max(theta - theta_ref)),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    params = {"nlev": 40, "dtheta": 2.0, "qv_max": 20.0,
              "zbubble": 5.0, "dt": 0.5, "nsteps": 1200}
    result = None
    if request.method == "POST":
        for k in params:
            v = request.form.get(k)
            params[k] = int(v) if k in ("nlev", "nsteps") else float(v)
        result = run_model_1d(**params)
    return render_template_string(
        HTML_1D,
        image=result["image"] if result else None,
        max_w=result["max_w"] if result else None,
        max_qv=result["max_qv"] if result else None,
        max_qc=result["max_qc"] if result else None,
        max_dtheta=result["max_dtheta"] if result else None,
        **params,
    )


# ====================== 2D ======================

def run_model_2d(nz, nx, x_total, dtheta, qv_max, zbubble, radius_x, dt, nsteps):
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

    n_snap = 8
    snap_every = max(1, nsteps // n_snap)
    times = [0.0]; w_hist = [w.copy()]; qc_hist = [qc.copy()]

    for n in range(nsteps):
        out = step_forward_2d(rho, w, vn, theta, qv, qc,
                              exner_ref_2d, init["theta_ref_2d"], g, dt)
        rho, w, vn = out["rho"], out["w"], out["vn"]
        theta, qv, qc = out["theta"], out["qv"], out["qc"]
        if (n + 1) % snap_every == 0 and len(times) < n_snap:
            times.append((n + 1) * dt)
            w_hist.append(w.copy()); qc_hist.append(qc.copy())

    image_b64 = make_snapshots_2d(w_hist, qc_hist,
                                  g["x_mc"], g["z_ifc"], g["z_mc"], times)
    return {
        "image": image_b64,
        "max_w": float(np.max(np.abs(w))),
        "max_vn": float(np.max(np.abs(vn))),
        "max_qc": float(np.max(qc)) * 1000,
    }


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
            params[k] = int(v) if k in ("nz", "nx", "nsteps") else float(v)
        result = run_model_2d(**params)
    return render_template_string(
        HTML_2D,
        image=result["image"] if result else None,
        max_w=result["max_w"] if result else None,
        max_vn=result["max_vn"] if result else None,
        max_qc=result["max_qc"] if result else None,
        **params,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)