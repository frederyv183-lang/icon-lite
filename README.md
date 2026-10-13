# icon-lite

Упрощённая 1D-модель атмосферы, собранная по уравнениям динамического ядра
ICON — негидростатической модели атмосферы, разработанной DWD и MPI-M.

Живая демонстрация: https://icon-lite.onrender.com

---

## Что это

Вертикальная колонка атмосферы, которая эволюционирует во времени
под действием:

- барического градиента (через возмущение функции Экспера pi'),
- плавучести (отклонение theta_v от опорного профиля),
- уравнения неразрывности (drho/dt = -d(rho*w)/dz),
- уравнения состояния pi = rho * R_d * theta_v / p0 (форма ICON).

Формулы извлечены вручную из исходников ICON, файл за файлом:
mo_solve_nonhydro.f90, mo_nonhydro_state.f90, mo_vertical_grid.f90.

---

## Что реализовано

| Компонент | Источник в ICON |
|---|---|
| Вертикальная сетка с растяжением | mo_vertical_grid.f90 |
| Опорная гидростатическая атмосфера | mo_nonhydro_state.f90 |
| Уравнение состояния | mo_solve_nonhydro.f90 |
| Возмущение Exner | строки 516-527 |
| Градиент Exner | строки 565-580 |
| Уравнение движения | блок 2180-2430 |
| Уравнение неразрывности | блок 2129-2153 |

Все формулы проверены численно — см. папку tests/.

---

## Запуск локально

    git clone https://github.com/frederyv183-lang/icon-lite.git
    cd icon-lite
    pip install -r requirements.txt
    python app.py

Откройте http://localhost:5000

---

## Тесты

    python tests/test_grid.py
    python tests/test_state.py
    python tests/test_exner.py
    python tests/test_motion.py
    python tests/test_continuity.py
    python tests/test_timestep.py
    python tests/test_plot.py

---

## Структура

    icon-lite/
      app.py           Flask-приложение
      src/
        grid.py        вертикальная сетка
        state.py       переменные и опорная атмосфера
        exner.py       градиенты Exner
        motion.py      уравнение движения
        continuity.py  уравнение неразрывности
        timestep.py    шаг по времени
        plotting.py    визуализация
      tests/           численные тесты

---

## Лицензия

MIT — см. LICENSE.

---

## Благодарности

Вдохновлено открытым исходным кодом ICON (BSD-3-Clause).
Учебная реализация, не претендующая на точное воспроизведение всех
процессов ICON.