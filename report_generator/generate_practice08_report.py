#!/usr/bin/env python3
"""
Генерация DOCX для практики №8 (вариант 5).

  python report_generator/generate_practice08_report.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from neuro_math_Yakimov_var5.task_8.practice08 import run_practice08  # noqa: E402
from report_generator.practice08_docx import build_practice08_docx  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="DOCX для практики №8")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs") / "practice08")
    parser.add_argument("--docx-name", type=str, default="Отчет_практическая_8_вариант_5.docx")
    parser.add_argument("--no-plots", action="store_true", help="Не перестраивать PNG")
    args = parser.parse_args()

    data = run_practice08(out_dir=args.out_dir, save_png=not args.no_plots)
    od = args.out_dir
    docx_path = od / args.docx_name

    build_practice08_docx(
        docx_path,
        data,
        plot_task1_bar=od / "practice08_task1_val_accuracy_char.png"
        if (od / "practice08_task1_val_accuracy_char.png").exists()
        else None,
    )
    print(f"Отчёт сохранён: {docx_path.resolve()}")


if __name__ == "__main__":
    main()
