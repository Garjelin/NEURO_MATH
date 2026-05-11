#!/usr/bin/env python3
"""
Генерация DOCX для практики №6.

Запуск из корня репозитория:
  python report_generator/generate_practice06_report.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from neuro_math_Yakimov_var5.task_6.practice06 import run_practice06  # noqa: E402
from report_generator.practice06_docx import build_practice06_docx  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="DOCX для практики №6")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs") / "practice06")
    parser.add_argument("--docx-name", type=str, default="Отчет_практическая_6_вариант_5.docx")
    parser.add_argument("--no-plots", action="store_true", help="Не перестраивать PNG")
    parser.add_argument("--epochs", type=int, default=1200, help="Эпохи NumPy-Adam на каждый dropout")
    args = parser.parse_args()

    data = run_practice06(out_dir=args.out_dir, save_png=not args.no_plots, epochs=args.epochs)
    p1 = args.out_dir / "practice06_dropout_test_forecasts.png"
    p2 = args.out_dir / "practice06_dropout_test_mse_bar.png"
    docx_path = args.out_dir / args.docx_name

    build_practice06_docx(
        docx_path,
        data,
        plot_forecasts=p1 if p1.exists() else None,
        plot_mse_bar=p2 if p2.exists() else None,
    )
    print(f"Отчёт сохранён: {docx_path.resolve()}")


if __name__ == "__main__":
    main()
