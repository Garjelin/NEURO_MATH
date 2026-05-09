#!/usr/bin/env python3
"""
Генерация DOCX для практики №4 (вариант 5).

Запуск из корня репозитория:
  python report_generator/generate_practice04_report.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from neuro_math_Yakimov_var5.task_4.practice04 import run_practice04  # noqa: E402
from report_generator.practice04_docx import build_practice04_docx  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="DOCX для практики №4")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs") / "practice04")
    parser.add_argument("--docx-name", type=str, default="Отчет_практическая_4_вариант_5.docx")
    parser.add_argument("--no-plots", action="store_true", help="Не перестраивать PNG")
    args = parser.parse_args()

    data = run_practice04(out_dir=args.out_dir, save_png=not args.no_plots)

    p1 = args.out_dir / "practice04_task1_tree.png"
    p2 = args.out_dir / "practice04_task1_predictions.png"
    p3 = args.out_dir / "practice04_task2_tree.png"
    p4 = args.out_dir / "practice04_task2_test_bar.png"
    docx_path = args.out_dir / args.docx_name

    build_practice04_docx(
        docx_path,
        data,
        plot_task1_tree=p1 if p1.exists() else None,
        plot_task1_pred=p2 if p2.exists() else None,
        plot_task2_tree=p3 if p3.exists() else None,
        plot_task2_bar=p4 if p4.exists() else None,
    )
    print(f"Отчёт сохранён: {docx_path.resolve()}")


if __name__ == "__main__":
    main()
