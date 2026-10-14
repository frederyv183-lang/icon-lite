"""
2D-турбулентность по Смагоринскому с подавлением по числу Ричардсона.

Отличия от 1D:
  - Деформация S учитывает горизонтальный сдвиг d(vn)/dx и смешанный член
  - Delta = sqrt(dx * dz) — эффективный размер ячейки
  - K_m — поле (nz, nx), а не профиль (nz,)
  - Диффузия применяется и по x, и по z
"""
import numpy as np

G = 9.81
CS = 0.15       # константа Смагоринского
RIC = 0.25      # критическое число Ричардсона
K_MIN = 0.0
K_MAX = 500.0   # м^2/с — защита от разлёта


def deformation_2d(w, vn, inv_dz, dz_ifc, dx_c, dx_ifc):
    """
    Скорость деформации |S| в центрах ячеек (nz, nx).

    S^2 = (dw/dz)^2 + (dvn/dx)^2 + 0.5 * (dw/dx + dvn/dz)^2

    Параметры
    ---------
    w : (nz+1, nx)      вертикальная скорость на z-интерфейсах
    vn : (nz, nx+1)     горизонтальная скорость на x-гранях
    inv_dz : (nz,)      1/dz (толщины центров)
    dz_ifc : (nz+1,)    расстояния между центрами по z
    dx_c : (nx,)        ширины центров (расстояние между гранями по x)
    dx_ifc : (nx+1,)    расстояния между центрами по x

    Возвращает
    ----------
    S : (nz, nx)        модуль скорости деформации (1/с)
    """
    nz = w.shape[0] - 1
    nx = w.shape[1]

    # dw/dz в центрах (nz, nx)
    dw_dz = (w[1:, :] - w[:-1, :]) * inv_dz[:, None]

    # dvn/dx в центрах — расстояние между гранями vn[:, i] и vn[:, i+1] = dx_c[i]
    dvn_dx = (vn[:, 1:] - vn[:, :-1]) / dx_c[None, :]

    # dw/dx в центрах — берём w на «нижнем» интерфейсе каждой ячейки
    w_c = w[1:, :]                                # (nz, nx)
    dw_dx = np.zeros((nz, nx))
    if nx > 2:
        dw_dx[:, 1:-1] = (w_c[:, 2:] - w_c[:, :-2]) / \
                         (dx_ifc[2:-1] + dx_ifc[1:-2])[None, :]
    dw_dx[:, 0] = (w_c[:, 1] - w_c[:, 0]) / dx_ifc[1]
    dw_dx[:, -1] = (w_c[:, -1] - w_c[:, -2]) / dx_ifc[-2]

    # dvn/dz на x-гранях (nz, nx+1) — центральная разность по z
    dvn_dz = np.zeros((nz, nx + 1))
    if nz > 2:
        # для k=1..nz-2: расстояние между центрами k-1 и k+1 = dz_ifc[k] + dz_ifc[k+1]
        dz_center = dz_ifc[1:-2] + dz_ifc[2:-1]   # (nz-2,)
        dvn_dz[1:-1, :] = (vn[2:, :] - vn[:-2, :]) / dz_center[:, None]
    dvn_dz[0, :] = (vn[1, :] - vn[0, :]) / dz_ifc[0]
    dvn_dz[-1, :] = (vn[-1, :] - vn[-2, :]) / dz_ifc[-1]

    # смешанный член: dw/dx + dvn/dz в центрах (nz, nx)
    dvndz_mc = 0.5 * (dvn_dz[:, :-1] + dvn_dz[:, 1:])  # (nz, nx)

    S2 = dw_dz ** 2 + dvn_dx ** 2 + 0.5 * (dw_dx + dvndz_mc) ** 2
    return np.sqrt(S2 + 1e-12)


def richardson_number_2d(theta_v, w, vn, inv_dz, dz_ifc, dx_c, dx_ifc):
    """
    Число Ричардсона Ri = N^2 / |S|^2 в центрах (nz, nx).
    N^2 = (g / theta_v) * d(theta_v)/dz
    """
    nz, nx = theta_v.shape

    dtheta_dz = np.zeros_like(theta_v)
    if nz > 2:
        dz_center = dz_ifc[1:-2] + dz_ifc[2:-1]   # (nz-2,)
        dtheta_dz[1:-1] = (theta_v[2:] - theta_v[:-2]) / dz_center[:, None]
    dtheta_dz[0] = (theta_v[1] - theta_v[0]) / dz_ifc[0]
    dtheta_dz[-1] = (theta_v[-1] - theta_v[-2]) / dz_ifc[-1]

    N2 = (G / theta_v) * dtheta_dz
    S = deformation_2d(w, vn, inv_dz, dz_ifc, dx_c, dx_ifc)
    S2 = S ** 2 + 1e-12

    Ri = N2 / S2
    return np.clip(Ri, -10.0, 10.0)


def km_smagorinsky_2d(theta_v, w, vn, inv_dz, dz, dz_ifc, dx, dx_ifc,
                      cs=CS, ric=RIC):
    """
    K_m в центрах (nz, nx). Delta = sqrt(dx * dz).
    """
    nz, nx = theta_v.shape

    Delta = np.sqrt(dz[:, None] * dx[None, :])       # (nz, nx)

    S = deformation_2d(w, vn, inv_dz, dz_ifc, dx, dx_ifc)
    Ri = richardson_number_2d(theta_v, w, vn, inv_dz, dz_ifc, dx, dx_ifc)

    F = np.maximum(0.0, 1.0 - Ri / ric)
    Km = (cs * Delta) ** 2 * S * F
    return np.clip(Km, K_MIN, K_MAX)


def _interp_to_z_ifc(field):
    """(nz, nx) -> (nz+1, nx), линейная интерполяция по z."""
    nz, nx = field.shape
    out = np.zeros((nz + 1, nx))
    if nz > 1:
        out[1:-1, :] = 0.5 * (field[:-1, :] + field[1:, :])
    out[0, :] = field[0, :]
    out[-1, :] = field[-1, :]
    return out


def _interp_to_x_edge(field):
    """(nz, nx) -> (nz, nx+1), линейная интерполяция по x."""
    nz, nx = field.shape
    out = np.zeros((nz, nx + 1))
    if nx > 1:
        out[:, 1:-1] = 0.5 * (field[:, :-1] + field[:, 1:])
    out[:, 0] = field[:, 0]
    out[:, -1] = field[:, -1]
    return out


def diffusion_step_2d(field, Km, dz, dz_ifc, dx, dx_ifc, dt):
    """
    Один шаг диффузии для поля (nz, nx):
        d(field)/dt = d/dx(Km * d(field)/dx) + d/dz(Km * d(field)/dz)
    """
    nz, nx = field.shape

    # Вертикальная часть
    Km_z = _interp_to_z_ifc(Km)                     # (nz+1, nx)
    dfield_dz = np.zeros((nz + 1, nx))
    if nz > 1:
        dfield_dz[1:-1, :] = (field[1:, :] - field[:-1, :]) / dz_ifc[1:-1, None]
    flux_z = Km_z * dfield_dz
    div_z = (flux_z[1:, :] - flux_z[:-1, :]) / dz[:, None]

    # Горизонтальная часть
    Km_x = _interp_to_x_edge(Km)                    # (nz, nx+1)
    dfield_dx = np.zeros((nz, nx + 1))
    if nx > 1:
        dfield_dx[:, 1:-1] = (field[:, 1:] - field[:, :-1]) / dx_ifc[1:-1][None, :]
    flux_x = Km_x * dfield_dx
    div_x = (flux_x[:, 1:] - flux_x[:, :-1]) / dx[None, :]

    return field + dt * (div_x + div_z)