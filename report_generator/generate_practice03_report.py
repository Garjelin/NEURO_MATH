#!/usr/bin/env python3
"""
Генерация DOCX для практики №3 (вариант 5).

Запуск:
  python report_generator/generate_practice03_report.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from neuro_math_Yakimov_var5.task_3.practice03 import run_practice03  # noqa: E402
from report_generator.practice03_docx import build_practice03_docx  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="DOCX для практики №3")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs") / "practice03")
    parser.add_argument("--docx-name", type=str, default="Отчет_практическая_3_вариант_5.docx")
    parser.add_argument("--no-plots", action="store_true", help="Не перестраивать PNG")
    args = parser.parse_args()

    data = run_practice03(out_dir=args.out_dir, save_png=not args.no_plots)
    p1 = args.out_dir / "practice03_scaling_demo.png"
    p2 = args.out_dir / "practice03_test_comparison.png"
    docx_path = args.out_dir / args.docx_name

    build_practice03_docx(
        docx_path,
        data,
        plot_scaling=p1 if p1.exists() else None,
        plot_test=p2 if p2.exists() else None,
    )
    print(f"Отчёт сохранён: {docx_path.resolve()}")


if __name__ == "__main__":
    main()
