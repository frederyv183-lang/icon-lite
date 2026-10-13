import sys
sys.path.insert(0, "src")
import numpy as np
from grid import make_vertical_grid
from state import make_reference_atmosphere
from timestep import step_forward

grid = make_vertical_grid(nlev=40, z_top=20000.0, stretch=1.15)
z_mc = grid["z_mc"]
z_ifc = grid["z_ifc"]
wgtfac_c = grid["wgtfac_c"]
inv_dz = 1.0 / grid["dz"]
dz = grid["dz"]

ref = make_reference_atmosphere(z_mc, z_ifc, theta_surf=300.0)

def run(use_diffusion, label):
    rho = ref["rho_ref"].copy()
    theta_ref = ref["theta_ref"]
    exner_ref = ref["exner_ref"]
    shape = np.exp(-((z_mc - 5000.0) / 1500.0) ** 2)
    theta = theta_ref + 2.0 * shape
    qv = 0.020 * shape
    qc = np.zeros_like(theta)
    w = np.zeros(len(z_ifc))

    for n in range(300):
        out = step_forward(rho, w, theta, qv, qc, exner_ref, theta_ref,
                           wgtfac_c, inv_dz, dz, z_mc, z_ifc, dt=0.5,
                           use_diffusion=use_diffusion)
        rho = out["rho"]; w = out["w"]
        theta = out["theta"]; qv = out["qv"]; qc = out["qc"]

    print(f'=== {label} ===')
    print(f'  max|w|  = {np.max(np.abs(w)):.4f} м/с')
    print(f'  max qv  = {np.max(qv)*1000:.3f} г/кг')
    print(f'  max qc  = {np.max(qc)*1000:.3f} г/кг')
    print(f'  max dθ  = {np.max(theta - theta_ref):.3f} K')
    return np.max(np.abs(w))

w_no = run(False, "БЕЗ турбулентности")
print()
w_yes = run(True, "С турбулентностью")
print()
print(f'Отношение max|w| (с / без): {w_yes / w_no:.4f}')