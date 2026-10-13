"""Запускает все test_*.py скрипты в tests/ как подпроцессы.

Это временная мера: большинство тестов написаны как скрипты
(код на уровне модуля, без def test_*). Pytest их не собирает.

Здесь мы прогоняем каждый скрипт через subprocess и проверяем,
что exit code = 0. Если скрипт падает — тест валится.

На Windows stdout по умолчанию в CP1251, и скрипты падают на
print() с Unicode-символами (→, θ, ²). Передаём PYTHONIOENCODING
и PYTHONUTF8, чтобы Python писал в UTF-8.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest


TESTS_DIR = Path(__file__).parent
REPO_ROOT = TESTS_DIR.parent
SKIP = {"test_all_scripts.py"}


def _collect_scripts():
    """Все test_*.py в tests/, кроме самой обёртки."""
    return sorted(
        p for p in TESTS_DIR.glob("test_*.py")
        if p.name not in SKIP
    )


def _make_env():
    """Окружение с UTF-8 для stdout/stderr дочернего Python."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


@pytest.mark.parametrize("script", _collect_scripts(), ids=lambda p: p.name)
def test_script_runs_without_error(script):
    """Запустить скрипт и убедиться, что exit code = 0."""
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        env=_make_env(),
    )
    if result.returncode != 0:
        msg = (
            f"\n{script.name} завершился с кодом {result.returncode}\n"
            f"--- STDOUT ---\n{result.stdout[-2000:]}\n"
            f"--- STDERR ---\n{result.stderr[-2000:]}\n"
        )
        pytest.fail(msg)