"""Tests for src/grid3d.py."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grid3d import make_grid3d


def test_grid3d_shapes_consistent():
    g = make_grid3d(nx=8, ny=6, nz=5, x_total=16000.0, y_total=12000.0)
    assert g["nx"] == 8
    assert g["ny"] == 6
    assert g["nz"] == 5
    assert g["x_mc"].shape == (8,)
    assert g["y_mc"].shape == (6,)
    assert g["z_mc"].shape == (5,)
    assert g["x_edge"].shape == (9,)
    assert g["y_edge"].shape == (7,)
    assert g["z_ifc"].shape == (6,)
    assert g["dx"].shape == (8,)
    assert g["dy"].shape == (6,)
    assert g["dz"].shape == (5,)


def test_grid3d_edges_and_centers():
    g = make_grid3d(nx=4, ny=4, nz=3, x_total=4000.0, y_total=4000.0)
    assert float(g["x_edge"][0]) == pytest.approx(0.0)
    assert float(g["x_edge"][-1]) == pytest.approx(4000.0)
    assert float(g["y_edge"][0]) == pytest.approx(0.0)
    assert float(g["y_edge"][-1]) == pytest.approx(4000.0)
    assert float(g["z_ifc"][0]) == pytest.approx(0.0)
    # x_mc in the middle of adjacent edges
    for i in range(4):
        mid = 0.5 * (g["x_edge"][i] + g["x_edge"][i + 1])
        assert float(g["x_mc"][i]) == pytest.approx(mid)
    for j in range(4):
        mid = 0.5 * (g["y_edge"][j] + g["y_edge"][j + 1])
        assert float(g["y_mc"][j]) == pytest.approx(mid)


def test_grid3d_z_stretch_and_wgtfac():
    g = make_grid3d(nx=4, ny=4, nz=10, z_top=20000.0, stretch=1.15)
    dz = g["dz"]
    # dz grows with height (geometric series)
    assert float(dz[1]) > float(dz[0])
    assert float(dz[-1]) > float(dz[-2])
    # sum of dz equals z_top
    assert float(np.sum(dz)) == pytest.approx(20000.0, rel=1e-9)
    # z_ifc monotonically increasing
    assert np.all(np.diff(g["z_ifc"]) > 0)
    # wgtfac_c in [0, 1]
    w = g["wgtfac_c"]
    assert np.all(w >= 0.0) and np.all(w <= 1.0)
    # z_mc = wgtfac_c * z_ifc[k] + (1 - wgtfac_c) * z_ifc[k+1]
    z_ifc = g["z_ifc"]; z_mc = g["z_mc"]
    recon = w * z_ifc[:-1] + (1.0 - w) * z_ifc[1:]
    assert np.allclose(recon, z_mc, atol=1e-9)
