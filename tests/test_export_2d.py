"""Тест CSV-экспорта 2D-поля (/download/2d)."""
import csv
import io
import sys
from pathlib import Path

import pytest

# Добавляем корень проекта в sys.path (как в других тестах)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def client():
    """Flask test client с минимальными параметрами."""
    import app as icon_app
    icon_app.app.config["TESTING"] = True
    with icon_app.app.test_client() as c:
        yield c


def test_download_2d_returns_csv(client):
    """GET /download/2d → 200 + Content-Type text/csv."""
    resp = client.get(
        "/download/2d?nz=6&nx=8&x_total=20.0&dtheta=2.0&qv_max=15.0"
        "&zbubble=3.0&radius_x=3.0&dt=0.5&nsteps=3"
    )
    assert resp.status_code == 200
    assert "text/csv" in resp.headers["Content-Type"]
    assert "icon_lite_2d.csv" in resp.headers["Content-Disposition"]


def test_download_2d_header_and_shape(client):
    """CSV: заголовок + nx*nz строк данных, 7 колонок."""
    nz, nx = 6, 8
    resp = client.get(
        f"/download/2d?nz={nz}&nx={nx}&x_total=20.0&dtheta=2.0&qv_max=15.0"
        "&zbubble=3.0&radius_x=3.0&dt=0.5&nsteps=3"
    )
    text = resp.data.decode("utf-8")
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)

    # 1 заголовок + nz*nx строк
    assert len(rows) == 1 + nz * nx

    # Заголовок
    assert rows[0] == ["x_km", "z_km", "w_ms", "theta_K",
                       "qv_kgkg", "qc_kgkg", "qr_kgkg"]

    # Каждая строка — 7 числовых полей
    for r in rows[1:]:
        assert len(r) == 7
        vals = [float(x) for x in r]
        assert all(v == v for v in vals)  # no NaN


def test_download_2d_ranges(client):
    """x_km ∈ [0, x_total], z_km ∈ [0, 20], qv/qc/qr ≥ 0."""
    nz, nx = 6, 8
    x_total = 20.0
    resp = client.get(
        f"/download/2d?nz={nz}&nx={nx}&x_total={x_total}&dtheta=2.0&qv_max=15.0"
        "&zbubble=3.0&radius_x=3.0&dt=0.5&nsteps=3"
    )
    reader = csv.DictReader(io.StringIO(resp.data.decode("utf-8")))
    for row in reader:
        x = float(row["x_km"])
        z = float(row["z_km"])
        assert 0.0 <= x <= x_total
        assert 0.0 <= z <= 20.0
        assert float(row["qv_kgkg"]) >= 0.0
        assert float(row["qc_kgkg"]) >= 0.0
        assert float(row["qr_kgkg"]) >= 0.0


def test_download_2d_empty_params_safe(client):
    """Пустые / невалидные параметры не должны ронять роут."""
    resp = client.get("/download/2d?nz=&nx=&dtheta=&dt=&nsteps=")
    # Роут должен отработать (или на дефолтах) без 500
    assert resp.status_code == 200