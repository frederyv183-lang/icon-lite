# icon-lite

[![tests](https://github.com/frederyv183-lang/icon-lite/actions/workflows/tests.yml/badge.svg)](https://github.com/frederyv183-lang/icon-lite/actions)
![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Упрощённая 1D- и 2D-модель атмосферной конвекции, собранная по уравнениям
динамического ядра ICON — негидростатической модели атмосферы,
разработанной DWD и MPI-M.

🌐 **Живая демонстрация:** https://icon-lite.onrender.com

| Страница | URL |
|---|---|
| 1D вертикальная колонка | https://icon-lite.onrender.com |
| 2D конвекция | https://icon-lite.onrender.com/2d |
| Сравнение 1D/2D | https://icon-lite.onrender.com/compare |
| Исходники | https://github.com/frederyv183-lang/icon-lite |

---

## Что это

Вертикальная колонка атмосферы, которая эволюционирует во времени
под действием:

- барического градиента (через возмущение функции Экспера pi'),
- плавучести (отклонение theta_v от опорного профиля),
- уравнения неразрывности (drho/dt = -d(rho*w)/dz),
- уравнения состояния pi = rho * R_d * theta_v / p0 (форма ICON).

Формулы извлечены вручную из исходников ICON, файл за файлом:
`mo_solve_nonhydro.f90`, `mo_nonhydro_state.f90`, `mo_vertical_grid.f90`.

---

## Версии

- **1D** (`/`) — вертикальная колонка с влажностью, конденсацией, осадками и турбулентностью.
- **2D** (`/2d`) — двумерная конвекция в плоскости (x, z):
  всплывающий тёплый пузырь, растекание, формирование облака, осадки.
- **Compare** (`/compare`) — сравнение 1D и 2D на одной странице.

---

## Physics

| Компонент | Источник в ICON |
|---|---|
| Вертикальная сетка с растяжением | `mo_vertical_grid.f90` |
| Опорная гидростатическая атмосфера | `mo_nonhydro_state.f90` |
| Уравнение состояния | `mo_solve_nonhydro.f90` |
| Возмущение Exner | строки 516–527 |
| Градиент Exner | строки 565–580 |
| Уравнение движения (w + vn) | блоки 2180–2430 |
| Уравнение неразрывности (1D + 2D) | блоки 2129–2153 |
| Влажность qv (трассер) | `mo_nh_moist_thdyn.f90` |
| Конденсация + скрытое тепло | `mo_nh_moist_thdyn.f90` |
| Осадки (qr): автоконверсия, осаждение, испарение | `mo_nh_moist_thdyn.f90` |
| Турбулентность (Смагоринский + Ri) | `mo_nh_diffusion.f90` |

Все формулы проверены численно — см. папку `tests/`.

---

## Validation

### 1D (вертикальная колонка, 40 уровней, dt = 0.5 с)

| Сценарий | max\|w\|, м/с | max qc, г/кг | max qr, г/кг |
|---|---|---|---|
| Сухой пузырь | 1.4 | — | — |
| Влажный пузырь | 3.9 | — | — |
| С конденсацией | 4.6 | 0.12 | 0.0 |
| С турбулентностью | 4.6 | 0.055 | 0.0 |
| dθ=3, qv=25 (с дождём) | 8.5 | 1.66 | 0.29 |

### 2D (сетка 40 × 80, dt = 0.5 с, 800 шагов)

| Величина | Значение |
|---|---|
| max\|w\| | 17.8 м/с |
| max\|vn\| | 12.5 м/с |
| max qc | 2.65 г/кг |

Формируется грибообразная структура с вихрями по бокам.

---

## Limitations

Проект **осознанно упрощён** относительно полного ICON:

- **Турбулентность не применяется к w** — только к `theta`, `qv`, `qc`.
  Сделано намеренно: `w` на интерфейсах, требует отдельной схемы.
- **Жёсткая крышка на 20 км** отражает волны. В ICON используется
  Rayleigh damping в верхних слоях.
- **2D без турбулентности** — не встроена.
- **Формула Тетенса для qsat** упрощённая (над водой, не над льдом).
- **L_v = 2.5e6** — постоянная, без температурной зависимости.
- **Осадки — 1D и 2D.** В 2D-модели `qr` рисуется синими пунктирными контурами.
- **qsat аппроксимация** для испарения `qr` использует порог 15 г/кг,
  а не полную формулу Тетенса.

---

## Roadmap

- [x] 1D модель с влажностью и конденсацией
- [x] 2D модель с горизонтальным растеканием
- [x] Турбулентность Смагоринского (1D)
- [x] Экспорт CSV (1D)
- [x] Страница сравнения 1D/2D
- [x] CI через GitHub Actions
- [x] Осадки (qr) — автоконверсия, осаждение, испарение
- [x] Осадки в 2D
- [ ] Валидация против настоящего ICON
- [ ] 3D-расширение

---

## Запуск локально

    git clone https://github.com/frederyv183-lang/icon-lite.git
    cd icon-lite
    pip install -r requirements.txt
    python app.py

Откройте http://localhost:5000

---

## Тесты

    python -m pytest tests/ -v

Все 23 теста должны пройти за ~12 сек.

Отдельные скрипты можно запускать вручную:

    python tests/test_grid.py
    python tests/test_state.py
    python tests/test_exner.py
    python tests/test_motion.py
    python tests/test_continuity.py
    python tests/test_timestep.py

---

## Структура

    icon-lite/
      app.py                  Flask-приложение (1D + 2D + compare)
      requirements.txt        numpy, matplotlib, flask, gunicorn, pytest
      Procfile                web: gunicorn app:app --bind 0.0.0.0:$PORT
      runtime.txt             python-3.12.0
      README.md               этот файл
      LICENSE                 MIT
      .github/workflows/
        tests.yml             CI: pytest при push

      src/
        grid.py               вертикальная сетка + интерполяция
        grid2d.py             2D-сетка (x, z)
        state.py              переменные + опорная атмосфера
        state2d.py            2D-инициализация (гауссов пузырь)
        exner.py              1D-градиенты Exner
        exner2d.py            2D-градиенты Exner
        thermo.py             виртуальная theta_v
        saturation.py         qsat (Тетенс) + saturation_adjustment
        motion.py             dw/dt + dvn/dt
        continuity.py         1D-неразрывность
        continuity2d.py       2D-неразрывность
        tracer.py             перенос qv
        precipitation.py      осадки: автоконверсия qc->qr, осаждение, испарение
        diffusion.py          турбулентность (Смагоринский)
        timestep.py           1D-шаг по времени
        timestep2d.py         2D-шаг по времени
        plotting.py           1D-визуализация
        plotting2d.py         2D-визуализация
        animation.py          1D-анимация кадров

      tests/                  23 теста (pytest + скрипты)

---

## Лицензия

MIT — см. LICENSE.

---

## Благодарности

Вдохновлено открытым исходным кодом ICON (BSD-3-Clause).
Учебная реализация, не претендующая на точное воспроизведение всех
процессов ICON.