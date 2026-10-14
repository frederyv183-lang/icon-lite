"""Tests for src/pressure3d.py (collocation, w on centres)."""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grid3d import make_grid3d
from pressure3d import project_divergence_free


def test_projection_reduces_divergence():
    g = make_grid3d(nx=12, ny=12, nz=12,
                    x_total=12000.0, y_total=12000.0,
                    z_top=12000.0, stretch=1.0)
    rng = np.random.default_rng(0)
    nz, ny, nx = g["nz"], g["ny"], g["nx"]
    u = rng.standard_normal((nz, ny, nx)) * 0.5
    v = rng.standard_normal((nz, ny, nx)) * 0.5
    w = rng.standard_normal((nz, ny, nx)) * 0.5

    u2, v2, w2, phi, div_b, div_a = project_divergence_free(u, v, w, g, dt=1.0)
    assert div_a < div_b, "divergence must not increase"
    assert div_a < 0.5 * div_b, "divergence should drop at least 2x"


def test_projection_zero_velocity_stays_zero():
    g = make_grid3d(nx=8, ny=8, nz=8,
                    x_total=8000.0, y_total=8000.0,
                    z_top=8000.0, stretch=1.0)
    nz, ny, nx = g["nz"], g["ny"], g["nx"]
    u = np.zeros((nz, ny, nx))
    v = np.zeros((nz, ny, nx))
    w = np.zeros((nz, ny, nx))

    u2, v2, w2, phi, div_b, div_a = project_divergence_free(u, v, w, g, dt=1.0)
    assert np.allclose(u2, 0.0, atol=1e-10)
    assert np.allclose(v2, 0.0, atol=1e-10)
    assert np.allclose(w2, 0.0, atol=1e-10)
    assert div_b < 1e-12
    assert div_a < 1e-12


def test_projection_shapes():
    g = make_grid3d(nx=10, ny=6, nz=8,
                    x_total=10000.0, y_total=6000.0,
                    z_top=8000.0, stretch=1.0)
    nz, ny, nx = g["nz"], g["ny"], g["nx"]
    u = np.zeros((nz, ny, nx))
    v = np.zeros((nz, ny, nx))
    w = np.ones((nz, ny, nx)) * 0.1
    u2, v2, w2, phi, _, _ = project_divergence_free(u, v, w, g, dt=1.0)
    assert u2.shape == (nz, ny, nx)
    assert v2.shape == (nz, ny, nx)
    assert w2.shape == (nz, ny, nx)
    assert phi.shape == (nz, ny, nx)
