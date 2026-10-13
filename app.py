"""
Веб-приложение icon-lite: запускает 1D-модель и показывает графики.
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


HTML = """
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
<h1>icon-lite — 1D-модель атмосферы</h1>
<div class="panel">
  <p>Упрощённая модель вертикальной колонки атмосферы,
     собранная по уравнениям динамического ядра ICON.</p>
  <form method="post">
    <label>Число уровней: <input type="number" name="nlev" value="{{ nlev }}" min="10" max="100"></label>
    <label>Амплитуда θ-возмущения, К: <input type="number" name="dtheta" value="{{ dtheta }}" step="0.5" min="0" max="20"></label>
    <label>Высота пузыря, км: <input type="number" name="zbubble" value="{{ zbubble }}" step="0.5" min="1" max="15"></label>
    <label>Шаг dt, с: <input type="number" name="dt" value="{{ dt }}" step="0.1" min="0.05" max="2"></label>
    <label>Число шагов: <input type="number" name="nsteps" value="{{ nsteps }}" min="100" max="5000"></label>
    <br><br>
    <button type="submit">Запустить модель</button>
  </form>
</div>

{% if image %}
<div class="panel">
  <h2>Результат</h2>
  <p class="meta">Финальный max|w| = {{ max_w }} м/с</p>
  <img src="data:image/png;base64,{{ image }}" alt="Профили">
</div>
{% endif %}
</body>
</html>
"""


def run_model(nlev, dtheta, zbubble, dt, nsteps):
    """Прогон модели и возврат PNG в base64."""
    grid = make_vertical_grid(nlev=nlev, z_top=20000.0, stretch=1.15)
    z_mc = grid["z_mc"]
    z_ifc = grid["z_ifc"]
    wgtfac_c = grid["wgtfac_c"]
    inv_dz = 1.0 / grid["dz"]

    ref = make_reference_atmosphere(z_mc, z_ifc, theta_surf=300.0)
    rho = ref["rho_ref"].copy()
    theta_ref = ref["theta_ref"]
    exner_ref = ref["exner_ref"]

    # тёплый пузырь
    theta_v = theta_ref + dtheta * np.exp(
        -((z_mc - zbubble * 1000.0) / 1500.0) ** 2
    )
    w = np.zeros(len(z_ifc))
    pi_pr = np.zeros_like(rho)

    # снимки
    n_snap = 4
    snap_every = max(1, nsteps // n_snap)
    times, w_hist, rho_hist, pi_hist = [], [], [], []
    times.append(0)
    w_hist.append(w.copy()); rho_hist.append(rho.copy()); pi_hist.append(pi_pr.copy())

    for n in range(nsteps):
        out = step_forward(rho, w, theta_v, exner_ref, theta_ref,
                           wgtfac_c, inv_dz, dt)
        rho = out["rho"]; w = out["w"]; pi_pr = out["pi_pr"]
        if (n + 1) % snap_every == 0 and len(times) < n_snap:
            times.append((n + 1) * dt)
            w_hist.append(w.copy()); rho_hist.append(rho.copy())
            pi_hist.append(pi_pr.copy())

    # графики
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(times)))

    ax = axes[0]
    for i, t in enumerate(times):
        ax.plot(w_hist[i], z_ifc / 1000.0, color=colors[i], label=f"t={t:.0f} с")
    ax.set_xlabel("w, м/с"); ax.set_ylabel("z, км")
    ax.set_title("Вертикальная скорость"); ax.axvline(0, color="k", lw=0.5, alpha=0.3)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    for i, t in enumerate(times):
        ax.plot(rho_hist[i], z_mc / 1000.0, color=colors[i])
    ax.set_xlabel("ρ, кг/м³"); ax.set_ylabel("z, км")
    ax.set_title("Плотность"); ax.grid(True, alpha=0.3)

    ax = axes[2]
    for i, t in enumerate(times):
        ax.plot(pi_hist[i] * 1000, z_mc / 1000.0, color=colors[i])
    ax.set_xlabel("π' × 1000"); ax.set_ylabel("z, км")
    ax.set_title("Возмущение Exner"); ax.axvline(0, color="k", lw=0.5, alpha=0.3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    image_b64 = base64.b64encode(buf.read()).decode("ascii")

    return image_b64, float(np.max(np.abs(w)))


@app.route("/", methods=["GET", "POST"])
def index():
    params = {
        "nlev": 40, "dtheta": 2.0, "zbubble": 5.0, "dt": 0.5, "nsteps": 1200,
    }
    image = None
    max_w = None

    if request.method == "POST":
        params["nlev"] = int(request.form.get("nlev", 40))
        params["dtheta"] = float(request.form.get("dtheta", 2.0))
        params["zbubble"] = float(request.form.get("zbubble", 5.0))
        params["dt"] = float(request.form.get("dt", 0.5))
        params["nsteps"] = int(request.form.get("nsteps", 1200))

        image, max_w = run_model(**params)

    return render_template_string(HTML, image=image, max_w=max_w, **params)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
