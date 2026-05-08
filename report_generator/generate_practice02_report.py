#!/usr/bin/env python3
"""
Генерация DOCX для практики №2 (отдельно от neuro_math_Yakimov_var5/task_2/practice02.py).

  python report_generator/generate_practice02_report.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from neuro_math_Yakimov_var5.task_2.practice02 import run_practice02  # noqa: E402
from report_generator.practice02_docx import build_practice02_docx  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="DOCX для практики №2")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs") / "practice02")
    parser.add_argument("--docx-name", type=str, default="Отчет_практическая_2_вариант_5.docx")
    parser.add_argument("--no-plots", action="store_true", help="Не перестраивать PNG")
    args = parser.parse_args()

    data = run_practice02(out_dir=args.out_dir, save_png=not args.no_plots)
    p1 = args.out_dir / "practice02_task1_fit.png"
    p2 = args.out_dir / "practice02_task2_forecast.png"
    docx_path = args.out_dir / args.docx_name
    build_practice02_docx(
        docx_path,
        data,
        plot_task1=p1 if p1.exists() else None,
        plot_task2=p2 if p2.exists() else None,
    )
    print(f"Отчёт сохранён: {docx_path.resolve()}")


if __name__ == "__main__":
    main()
