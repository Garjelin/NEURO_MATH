#!/usr/bin/env python3
"""
Генерация DOCX для практики №9 (вариант 5).

  python report_generator/generate_practice09_report.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from neuro_math_Yakimov_var5.task_9.practice09 import run_practice09  # noqa: E402
from report_generator.practice09_docx import build_practice09_docx  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="DOCX для практики №9")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs") / "practice09")
    parser.add_argument("--docx-name", type=str, default="Отчет_практическая_9_вариант_5.docx")
    parser.add_argument("--no-plots", action="store_true", help="Не перестраивать PNG")
    args = parser.parse_args()

    data = run_practice09(out_dir=args.out_dir, save_png=not args.no_plots)
    od = args.out_dir
    docx_path = od / args.docx_name

    build_practice09_docx(
        docx_path,
        data,
        plot_samples=od / "practice09_style_samples.png" if (od / "practice09_style_samples.png").exists() else None,
        plot_confusion=od / "practice09_confusion_matrix.png"
        if (od / "practice09_confusion_matrix.png").exists()
        else None,
        plot_queries=od / "practice09_query_predictions.png"
        if (od / "practice09_query_predictions.png").exists()
        else None,
    )
    print(f"Отчёт сохранён: {docx_path.resolve()}")


if __name__ == "__main__":
    main()
