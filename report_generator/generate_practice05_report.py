#!/usr/bin/env python3
"""
Генерация DOCX для практики №5 (вариант 5).

Запуск из корня репозитория:
  python report_generator/generate_practice05_report.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from neuro_math_Yakimov_var5.task_5.practice05 import run_practice05  # noqa: E402
from report_generator.practice05_docx import build_practice05_docx  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="DOCX для практики №5")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs") / "practice05")
    parser.add_argument("--docx-name", type=str, default="Отчет_практическая_5_вариант_5.docx")
    parser.add_argument("--no-plots", action="store_true", help="Не перестраивать PNG")
    args = parser.parse_args()

    data = run_practice05(out_dir=args.out_dir, save_png=not args.no_plots)

    p1 = args.out_dir / "practice05_task1_boundary.png"
    p2 = args.out_dir / "practice05_task1_confusion_test.png"
    p3 = args.out_dir / "practice05_task2_series_tail.png"
    p4 = args.out_dir / "practice05_task2_confusion_test.png"
    docx_path = args.out_dir / args.docx_name

    build_practice05_docx(
        docx_path,
        data,
        plot_task1_boundary=p1 if p1.exists() else None,
        plot_task1_cm=p2 if p2.exists() else None,
        plot_task2_series=p3 if p3.exists() else None,
        plot_task2_cm=p4 if p4.exists() else None,
    )
    print(f"Отчёт сохранён: {docx_path.resolve()}")


if __name__ == "__main__":
    main()
