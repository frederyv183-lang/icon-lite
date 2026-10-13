"""
Веб-приложение icon-lite: 1D-модель атмосферы с конденсацией.
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
from state import make_reference_atmosphere
from timestep import step_forward

app = Flask(__name__)


HTML = '''
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>icon-lite</title>
<style>
  body { font-family: sans-serif; max-width: 1200px; margin: 2em auto;
         background: #f5f5f7; color: #222; }
  h1 { font-weight: 300; }
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
<h1>icon-lite — 1D-модель атмосферы с конденсацией</h1>
<div class="panel">
  <p>Модель вертикальной колонки атмосферы по уравнениям ICON.
     С учётом влажности и конденсации (скрытое тепло).
     Зелёная кривая — облачная вода qc — это облако.</p>
  <form method="post">
    <label>Число уровней: <input type="number" name="nlev" value="{{ nlev }}" min="10" max="100"></label>
    <label>θ-возмущение, К: <input type="number" name="dtheta" value="{{ dtheta }}" step="0.5" min="0" max="20"></label>
    <label>Влажность qv, г/кг: <input type="number" name="qv_max" value="{{ qv_max }}" step="1" min="0" max="30"></label>
    <label>Высота пузыря, км: <input type="number" name="zbubble" value="{{ zbubble }}" step="0.5" min="1" max="15"></label>
    <label>Шаг dt, с: <input type="number" name="dt" value="{{ dt }}" step="0.05" min="0.05" max="2"></label>
    <label>Число шагов: <input type="number" name="nsteps" value="{{ nsteps }}" min="100" max="5000"></label>
    <br><br>
    <button type="submit">Запустить модель</button>
  </form>
</div>

{% if image %}
<div class="panel">
  <h2>Эволюция конвективного облака</h2>
  <p class="meta">
    Финальный max|w| = {{ max_w }} м/с &nbsp;|&nbsp;
    max qv = {{ max_qv }} г/кг &nbsp;|&nbsp;
    max qc = {{ max_qc }} г/кг &nbsp;|&nbsp;
    max dθ = {{ max_dtheta }} K
  </p>
  <img src="data:image/png;base64,{{ image }}" alt="Эволюция">
</div>
{% endif %}
</body>
</html>
'''


def make_frames_frame(w_hist, qv_hist, qc_hist, z_ifc, z_mc, times,
                      ncols=4, figsize=(16, 8)):
    """
    Сетка кадров nrows x ncols: три профиля на каждом — w, qv, qc.
    """
    n = len(times)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    fig.subplots_adjust(hspace=0.55, wspace=0.4)

    for i in range(n):
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]

        # w (синий)
        ax.plot(w_hist[i], z_ifc / 1000.0, color="tab:blue", lw=1.2)
        ax.set_xlim(-8, 8)
        ax.set_ylim(0, 20)
        ax.axvline(0, color="k", lw=0.4, alpha=0.4)
        ax.set_title(f"t = {times[i]:.0f} с", fontsize=10)
        if c == 0:
            ax.set_ylabel("z, км", fontsize=9)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        if c == 0 and r == nrows - 1:
            ax.set_xlabel("w, м/с", fontsize=8, color="tab:blue")

        # qv (красный) на верхней оси
        ax2 = ax.twiny()
        ax2.plot(qv_hist[i] * 1000, z_mc / 1000.0, color="tab:red", lw=1.0,
                 alpha=0.85)
        ax2.set_xlim(0, 25)
        ax2.tick_params(labelsize=8, colors="tab:red")
        ax2.spines["top"].set_color("tab:red")
        if c == ncols - 1 and r == 0:
            ax2.set_xlabel("qv, г/кг", fontsize=8, color="tab:red")

        # qc (зелёный) — облачная вода
        ax3 = ax.twiny()
        ax3.spines["top"].set_position(("axes", 1.18))
        ax3.spines["top"].set_color("tab:green")
        ax3.plot(qc_hist[i] * 1000, z_mc / 1000.0, color="tab:green", lw=1.4)
        ax3.set_xlim(0, 5)
        ax3.tick_params(labelsize=7, colors="tab:green")
        if c == ncols - 1 and r == 0:
            ax3.set_xlabel("qc, г/кг", fontsize=8, color="tab:green")

    for i in range(n, nrows * ncols):
        r = i // ncols
        c = i % ncols
        axes[r, c].axis("off")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def run_model(nlev, dtheta, qv_max, zbubble, dt, nsteps):
    grid = make_vertical_grid(nlev=nlev, z_top=20000.0, stretch=1.15)
    z_mc = grid["z_mc"]
    z_ifc = grid["z_ifc"]
    wgtfac_c = grid["wgtfac_c"]
    inv_dz = 1.0 / grid["dz"]

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
    times = [0.0]
    w_hist = [w.copy()]
    qv_hist = [qv.copy()]
    qc_hist = [qc.copy()]

    for n in range(nsteps):
        out = step_forward(rho, w, theta, qv, qc, exner_ref, theta_ref,
                           wgtfac_c, inv_dz, dt)
        rho = out["rho"]; w = out["w"]
        theta = out["theta"]; qv = out["qv"]; qc = out["qc"]
        if (n + 1) % snap_every == 0 and len(times) < n_snap:
            times.append((n + 1) * dt)
            w_hist.append(w.copy())
            qv_hist.append(qv.copy())
            qc_hist.append(qc.copy())

    image_b64 = make_frames_frame(w_hist, qv_hist, qc_hist,
                                  z_ifc, z_mc, times)
    return {
        "image": image_b64,
        "max_w": float(np.max(np.abs(w))),
        "max_qv": float(np.max(qv)) * 1000,
        "max_qc": float(np.max(qc)) * 1000,
        "max_dtheta": float(np.max(theta - theta_ref)),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    params = {
        "nlev": 40, "dtheta": 2.0, "qv_max": 20.0,
        "zbubble": 5.0, "dt": 0.5, "nsteps": 1200,
    }
    result = None

    if request.method == "POST":
        params["nlev"] = int(request.form.get("nlev", 40))
        params["dtheta"] = float(request.form.get("dtheta", 2.0))
        params["qv_max"] = float(request.form.get("qv_max", 20.0))
        params["zbubble"] = float(request.form.get("zbubble", 5.0))
        params["dt"] = float(request.form.get("dt", 0.5))
        params["nsteps"] = int(request.form.get("nsteps", 1200))
        result = run_model(**params)

    return render_template_string(
        HTML,
        image=result["image"] if result else None,
        max_w=result["max_w"] if result else None,
        max_qv=result["max_qv"] if result else None,
        max_qc=result["max_qc"] if result else None,
        max_dtheta=result["max_dtheta"] if result else None,
        **params,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)